% Assignment 3, Computer Vision & Pattern Recognition
% 1430003045 ÷”æ˚»Â
% Image Stitching
% Code outside this file refers to: 
% http://blog.csdn.net/zx9105080016/article/details/41048477
% Problem: Output of RANSAC function is not stable. Sometime the program
% will fail.

% Read image
img1 = imread('1.jpg');
img2 = imread('2.jpg');
% Get size info
[height1,width1,~] = size(img1);
[height2,width2,~] = size(img2);
% Convert to gray
gray1 = im2double(rgb2gray(img1));
gray2 = im2double(rgb2gray(img2));

% Find Harris coeners of both images
[x_A,y_A,v_A] = harris(gray1,2,0.0,2);
[x_B,y_B,v_B] = harris(gray2,2,0.0,2);
% Apply Adaptive Non-maximal Suppression
ncorners = 500;
[x_A,y_A,~] = ada_nonmax_suppression(x_A,y_A,v_A,ncorners);
[x_B,y_B,~] = ada_nonmax_suppression(x_B,y_B,v_B,ncorners);
% Extract feature descrption
sigma = 7;
[des_A] = getFeatureDescriptor(gray1,x_A,y_A,sigma);
[des_B] = getFeatureDescriptor(gray2,x_B,y_B,sigma);

% Feature matching
dist = dist2(des_A,des_B);
[ord_dist,index] = sort(dist,2);
ratio = ord_dist(:,1)./ord_dist(:,2);
threshold = 0.5;
idx = ratio<threshold;

x_A = x_A(idx);
y_A = y_A(idx);
x_B = x_B(index(idx,1));
y_B = y_B(index(idx,1));
npoints = length(x_A);

% Use RANSAC to compute a estimate.
matcher_A = [y_A, x_A, ones(npoints,1)]'; 
matcher_B = [y_B, x_B, ones(npoints,1)]'; 
[hh, ~] = ransacfithomography(matcher_B, matcher_A, npoints, 10);

% Inverse warp.
% Set up the size of new image.
[newH,newW,newX,newY,xB,yB] = getNewSize(hh,height1,width1,height2,width2);

[X,Y] = meshgrid(1:width1,1:height1);
[XX,YY] = meshgrid(newX:newX+newW-1,newY:newY+newH-1);
newImg = ones(3,newH*newW);
newImg(1,:) = reshape(XX,1,newH*newW);
newImg(2,:) = reshape(YY,1,newH*newW);

newImg = hh * newImg;
XX = reshape(newImg(1,:)./newImg(3,:),newH,newW);
YY = reshape(newImg(2,:)./newImg(3,:),newH,newW);

% Interploation, do warpping.
newImage(:,:,1) = interp2(X, Y, double(img1(:,:,1)), XX, YY);
newImage(:,:,2) = interp2(X, Y, double(img1(:,:,2)), XX, YY);
newImage(:,:,3) = interp2(X, Y, double(img1(:,:,3)), XX, YY);

% Blend image by cross disslove.
[newImage] = blend(newImage,img2,xB,yB);

% Show the image
imshow(uint8(newImage));