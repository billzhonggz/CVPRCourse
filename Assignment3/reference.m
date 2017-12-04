function [output_image] = image_stitching(input_A, input_B)
% -------------------------------------------------------------------------
% 1. Load both images, convert to double and to grayscale.
% 2. Detect feature points in both images.
% 3. Extract fixed-size patches around every keypoint in both images, and
% form descriptors simply by "flattening" the pixel values in each patch to
% one-dimensional vectors.
% 4. Compute distances between every descriptor in one image and every descriptor in the other image. 
% 5. Select putative matches based on the matrix of pairwise descriptor
% distances obtained above. 
% 6. Run RANSAC to estimate (1) an affine transformation and (2) a
% homography mapping one image onto the other. 
% 7. Warp one image onto the other using the estimated transformation.
% 8. Create a new image big enough to hold the panorama and composite the
% two images into it. 
% 
% Input:
% input_A - filename of warped image
% input_B - filename of unwarped image
% Output:
% output_image - combined new image
% 
% Reference:
% [1] C.G. Harris and M.J. Stephens, A combined corner and edge detector, 1988.
% [2] Matthew Brown, Multi-Image Matching using Multi-Scale Oriented Patches.
% 
% zhyh8341@gmail.com

% -------------------------------------------------------------------------

% READ IMAGE, GET SIZE INFORMATION
image_A = imread(input_A);
image_B = imread(input_B);
[height_wrap, width_wrap,~] = size(image_A);
[height_unwrap, width_unwrap,~] = size(image_B);

% CONVERT TO GRAY SCALE
gray_A = im2double(rgb2gray(image_A));
gray_B = im2double(rgb2gray(image_B));


% FIND HARRIS CORNERS IN BOTH IMAGE
[x_A, y_A, v_A] = harris(gray_A, 2, 0.0, 2);
[x_B, y_B, v_B] = harris(gray_B, 2, 0.0, 2);

% ADAPTIVE NON-MAXIMAL SUPPRESSION (ANMS)
ncorners = 500;
[x_A, y_A, ~] = ada_nonmax_suppression(x_A, y_A, v_A, ncorners);
[x_B, y_B, ~] = ada_nonmax_suppression(x_B, y_B, v_B, ncorners);

% EXTRACT FEATURE DESCRIPTORS
sigma = 7;
[des_A] = getFeatureDescriptor(gray_A, x_A, y_A, sigma);
[des_B] = getFeatureDescriptor(gray_B, x_B, y_B, sigma);

% IMPLEMENT FEATURE MATCHING
dist = dist2(des_A,des_B);
[ord_dist, index] = sort(dist, 2);
% THE RATIO OF FIRST AND SECOND DISTANCE IS A BETTER CRETIA THAN DIRECTLY
% USING THE DISTANCE. RATIO LESS THAN .5 GIVES AN ACCEPTABLE ERROR RATE.
ratio = ord_dist(:,1)./ord_dist(:,2);
threshold = 0.5;
idx = ratio<threshold;

x_A = x_A(idx);
y_A = y_A(idx);
x_B = x_B(index(idx,1));
y_B = y_B(index(idx,1));
npoints = length(x_A);


% USE 4-POINT RANSAC TO COMPUTE A ROBUST HOMOGRAPHY ESTIMATE
% KEEP THE FIRST IMAGE UNWARPED, WARP THE SECOND TO THE FIRST
matcher_A = [y_A, x_A, ones(npoints,1)]'; %!!! previous x is y and y is x,
matcher_B = [y_B, x_B, ones(npoints,1)]'; %!!! so switch x and y here.
[hh, ~] = ransacfithomography(matcher_B, matcher_A, npoints, 10);

% s = load('matcher.mat');
% matcher_A = s.matcher(1:3,:);
% matcher_B = s.matcher(4:6,:);
% npoints = 60;
% [hh, inliers] = ransacfithomography(matcher_B, matcher_A, npoints, 10);


% USE INVERSE WARP METHOD
% DETERMINE THE SIZE OF THE WHOLE IMAGE
[newH, newW, newX, newY, xB, yB] = getNewSize(hh, height_wrap, width_wrap, height_unwrap, width_unwrap);

[X,Y] = meshgrid(1:width_wrap,1:height_wrap);
[XX,YY] = meshgrid(newX:newX+newW-1, newY:newY+newH-1);
AA = ones(3,newH*newW);
AA(1,:) = reshape(XX,1,newH*newW);
AA(2,:) = reshape(YY,1,newH*newW);

AA = hh*AA;
XX = reshape(AA(1,:)./AA(3,:), newH, newW);
YY = reshape(AA(2,:)./AA(3,:), newH, newW);

% INTERPOLATION, WARP IMAGE A INTO NEW IMAGE
newImage(:,:,1) = interp2(X, Y, double(image_A(:,:,1)), XX, YY);
newImage(:,:,2) = interp2(X, Y, double(image_A(:,:,2)), XX, YY);
newImage(:,:,3) = interp2(X, Y, double(image_A(:,:,3)), XX, YY);

% BLEND IMAGE BY CROSS DISSOLVE
[newImage] = blend(newImage, image_B, xB, yB);

% DISPLAY IMAGE MOSIAC
imshow(uint8(newImage));

 

% -------------------------------------------------------------------------
% ------------------------------- other functions -------------------------
% -------------------------------------------------------------------------
function [xp, yp, value] = harris(input_image, sigma,thd, r)
% Detect harris corner 
% Input:
% sigma - standard deviation of smoothing Gaussian
% r - radius of region considered in non-maximal suppression
% Output:
% xp - x coordinates of harris corner points
% yp - y coordinates of harris corner points
% value - values of R at harris corner points

% CONVERT RGB IMAGE TO GRAY-SCALE, AND BLUR WITH G1 KERNEL
g1 = fspecial('gaussian', 7, 1);
gray_image = imfilter(input_image, g1);

% FILTER INPUT IMAGE WITH SOBEL KERNEL TO GET GRADIENT ON X AND Y
% ORIENTATION RESPECTIVELY
h = fspecial('sobel');
Ix = imfilter(gray_image,h,'replicate','same');
Iy = imfilter(gray_image,h','replicate','same');

% GENERATE GAUSSIAN FILTER OF SIZE 6*SIGMA (± 3SIGMA) AND OF MINIMUM SIZE 1x1
g = fspecial('gaussian',fix(6*sigma), sigma);

Ix2 = imfilter(Ix.^2, g, 'same').*(sigma^2); 
Iy2 = imfilter(Iy.^2, g, 'same').*(sigma^2);
Ixy = imfilter(Ix.*Iy, g, 'same').*(sigma^2);

% HARRIS CORNER MEASURE
R = (Ix2.*Iy2 - Ixy.^2)./(Ix2 + Iy2 + eps); 
% ANOTHER MEASUREMENT, USUALLY k IS BETWEEN 0.04 ~ 0.06
% response = (Ix2.*Iy2 - Ixy.^2) - k*(Ix2 + Iy2).^2;

% GET RID OF CORNERS WHICH IS CLOSE TO BORDER
R([1:20, end-20:end], :) = 0;
R(:,[1:20,end-20:end]) = 0;

% SUPRESS NON-MAX 
d = 2*r+1; 
localmax = ordfilt2(R,d^2,true(d)); 
R = R.*(and(R==localmax, R>thd));

% RETURN X AND Y COORDINATES 
[xp,yp,value] = find(R);

function [newx, newy, newvalue] = ada_nonmax_suppression(xp, yp, value, n)
% Adaptive non-maximun suppression 
% For each Harris Corner point, the minimum suppression radius is the
% minimum distance from that point to a different point with a higher 
% corner strength. 
% Input:
% xp,yp - coordinates of harris corner points
% value - strength of suppression
% n - number of interesting points
% Output:
% newx, newy - new x and y coordinates after adaptive non-maximun suppression
% value - strength of suppression after adaptive non-maximun suppression

% ALLOCATE MEMORY
% newx = zeros(n,1);
% newy = zeros(n,1);
% newvalue = zeros(n,1);

if(length(xp) < n)
newx = xp;
newy = yp;
newvalue = value;
return;
end

radius = zeros(n,1);
c = .9;
maxvalue = max(value)*c;
for i=1:length(xp)
if(value(i)>maxvalue)
radius(i) = 99999999;
continue;
else
dist = (xp-xp(i)).^2 + (yp-yp(i)).^2;
dist((value*c) < value(i)) = [];
radius(i) = sqrt(min(dist));
end
end

[~, index] = sort(radius,'descend');
index = index(1:n);

newx = xp(index);
newy = yp(index);
newvalue = value(index);

function n2 = dist2(x, c)
% DIST2 Calculates squared distance between two sets of points.
% Adapted from Netlab neural network software:
% http://www.ncrg.aston.ac.uk/netlab/index.php
%
% Description
% D = DIST2(X, C) takes two matrices of vectors and calculates the
% squared Euclidean distance between them. Both matrices must be of
% the same column dimension. If X has M rows and N columns, and C has
% L rows and N columns, then the result has M rows and L columns. The
% I, Jth entry is the squared distance from the Ith row of X to the
% Jth row of C.
%
%
% Copyright (c) Ian T Nabney (1996-2001)

[ndata, dimx] = size(x);
[ncentres, dimc] = size(c);
if dimx ~= dimc
error('Data dimension does not match dimension of centres')
end

n2 = (ones(ncentres, 1) * sum((x.^2)', 1))' + ...
ones(ndata, 1) * sum((c.^2)',1) - ...
2.*(x*(c'));

% Rounding errors occasionally cause negative entries in n2
if any(any(n2<0))
n2(n2<0) = 0;
end

function [descriptors] = getFeatureDescriptor(input_image, xp, yp, sigma)
% Extract non-rotation invariant feature descriptors
% Input:
% input_image - input gray-scale image
% xx - x coordinates of potential feature points
% yy - y coordinates of potential feature points
% output:
% descriptors - array of descriptors

% FIRST BLUR WITH GAUSSIAN KERNEL
g = fspecial('gaussian', 5, sigma);
blurred_image = imfilter(input_image, g, 'replicate','same');

% THEN TAKE A 40x40 PIXEL WINDOW AND DOWNSAMPLE TO 8x8 PATCH
npoints = length(xp);
descriptors = zeros(npoints,64);

for i = 1:npoints
%pA = imresize( blurred_image(xp(i)-20:xp(i)+19, yp(i)-20:yp(i)+19), .2);
patch = blurred_image(xp(i)-20:xp(i)+19, yp(i)-20:yp(i)+19);
patch = imresize(patch, .2);
descriptors(i,:) = reshape((patch - mean2(patch))./std2(patch), 1, 64); 
end

function [hh] = getHomographyMatrix(point_ref, point_src, npoints)
% Use corresponding points in both images to recover the parameters of the transformation 
% Input:
% x_ref, x_src --- x coordinates of point correspondences
% y_ref, y_src --- y coordinates of point correspondences
% Output:
% h --- matrix of transformation

% NUMBER OF POINT CORRESPONDENCES
x_ref = point_ref(1,:)';
y_ref = point_ref(2,:)';
x_src = point_src(1,:)';
y_src = point_src(2,:)';

% COEFFICIENTS ON THE RIGHT SIDE OF LINEAR EQUATIONS
A = zeros(npoints*2,8);
A(1:2:end,1:3) = [x_ref, y_ref, ones(npoints,1)];
A(2:2:end,4:6) = [x_ref, y_ref, ones(npoints,1)];
A(1:2:end,7:8) = [-x_ref.*x_src, -y_ref.*x_src];
A(2:2:end,7:8) = [-x_ref.*y_src, -y_ref.*y_src];

% COEFFICIENT ON THE LEFT SIDE OF LINEAR EQUATIONS
B = [x_src, y_src];
B = reshape(B',npoints*2,1);

% SOLVE LINEAR EQUATIONS
h = A\B;

hh = [h(1),h(2),h(3);h(4),h(5),h(6);h(7),h(8),1];

function [hh, inliers] = ransacfithomography(ref_P, dst_P, npoints, threshold);
% 4-point RANSAC fitting
% Input:
% matcher_A - match points from image A, a matrix of 3xN, the third row is 1
% matcher_B - match points from image B, a matrix of 3xN, the third row is 1
% thd - distance threshold
% npoints - number of samples
% 
% 1. Randomly select minimal subset of points
% 2. Hypothesize a model
% 3. Computer error function
% 4. Select points consistent with model
% 5. Repeat hypothesize-and-verify loop
% 
% Yihua Zhao 02-01-2014
% zhyh8341@gmail.com

ninlier = 0;
fpoints = 8; %number of fitting points
for i=1:2000
rd = randi([1 npoints],1,fpoints);
pR = ref_P(:,rd);
pD = dst_P(:,rd);
h = getHomographyMatrix(pR,pD,fpoints);
rref_P = h*ref_P;
rref_P(1,:) = rref_P(1,:)./rref_P(3,:);
rref_P(2,:) = rref_P(2,:)./rref_P(3,:);
error = (rref_P(1,:) - dst_P(1,:)).^2 + (rref_P(2,:) - dst_P(2,:)).^2;
n = nnz(error<threshold);
if(n >= npoints*.95)
hh=h;
inliers = find(error<threshold);
pause();
break;
elseif(n>ninlier)
ninlier = n;
hh=h;
inliers = find(error<threshold);
end 
end

function [newH, newW, x1, y1, x2, y2] = getNewSize(transform, h2, w2, h1, w1)
% Calculate the size of new mosaic
% Input:
% transform - homography matrix
% h1 - height of the unwarped image
% w1 - width of the unwarped image
% h2 - height of the warped image
% w2 - height of the warped image
% Output:
% newH - height of the new image
% newW - width of the new image
% x1 - x coordate of lefttop corner of new image
% y1 - y coordate of lefttop corner of new image
% x2 - x coordate of lefttop corner of unwarped image
% y2 - y coordate of lefttop corner of unwarped image
% 
% Yihua Zhao 02-02-2014
% zhyh8341@gmail.com
%

% CREATE MESH-GRID FOR THE WARPED IMAGE
[X,Y] = meshgrid(1:w2,1:h2);
AA = ones(3,h2*w2);
AA(1,:) = reshape(X,1,h2*w2);
AA(2,:) = reshape(Y,1,h2*w2);

% DETERMINE THE FOUR CORNER OF NEW IMAGE
newAA = transform\AA;
new_left = fix(min([1,min(newAA(1,:)./newAA(3,:))]));
new_right = fix(max([w1,max(newAA(1,:)./newAA(3,:))]));
new_top = fix(min([1,min(newAA(2,:)./newAA(3,:))]));
new_bottom = fix(max([h1,max(newAA(2,:)./newAA(3,:))]));

newH = new_bottom - new_top + 1;
newW = new_right - new_left + 1;
x1 = new_left;
y1 = new_top;
x2 = 2 - new_left;
y2 = 2 - new_top;

function [newImage] = blend(warped_image, unwarped_image, x, y)
% Blend two image by using cross dissolve
% Input:
% warped_image - original image
% unwarped_image - the other image
% x - x coordinate of the lefttop corner of unwarped image
% y - y coordinate of the lefttop corner of unwarped image
% Output:
% newImage
% 
% Yihua Zhao 02-02-2014
% zhyh8341@gmail.com
%


% MAKE MASKS FOR BOTH IMAGES 
warped_image(isnan(warped_image))=0;
maskA = (warped_image(:,:,1)>0 |warped_image(:,:,2)>0 | warped_image(:,:,3)>0);
newImage = zeros(size(warped_image));
newImage(y:y+size(unwarped_image,1)-1, x: x+size(unwarped_image,2)-1,:) = unwarped_image;
mask = (newImage(:,:,1)>0 | newImage(:,:,2)>0 | newImage(:,:,3)>0);
mask = and(maskA, mask);

% GET THE OVERLAID REGION
[~,col] = find(mask);
left = min(col);
right = max(col);
mask = ones(size(mask));
if( x<2)
mask(:,left:right) = repmat(linspace(0,1,right-left+1),size(mask,1),1);
else
mask(:,left:right) = repmat(linspace(1,0,right-left+1),size(mask,1),1);
end

% BLEND EACH CHANNEL
warped_image(:,:,1) = warped_image(:,:,1).*mask;
warped_image(:,:,2) = warped_image(:,:,2).*mask;
warped_image(:,:,3) = warped_image(:,:,3).*mask;

% REVERSE THE ALPHA VALUE
if( x<2)
mask(:,left:right) = repmat(linspace(1,0,right-left+1),size(mask,1),1);
else
mask(:,left:right) = repmat(linspace(0,1,right-left+1),size(mask,1),1);
end
newImage(:,:,1) = newImage(:,:,1).*mask;
newImage(:,:,2) = newImage(:,:,2).*mask;
newImage(:,:,3) = newImage(:,:,3).*mask;

newImage(:,:,1) = warped_image(:,:,1) + newImage(:,:,1);
newImage(:,:,2) = warped_image(:,:,2) + newImage(:,:,2);
newImage(:,:,3) = warped_image(:,:,3) + newImage(:,:,3);