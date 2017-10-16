function [ ] = linTransformImg( image,k,m )
%Gray linear transformation.
%   1430003045 ÷”æ˚»Â

% Transfer to gray.
gray = rgb2gray(image);
% Display the original image.
figure(1), imshow(gray);
title('Original image');
% Display the original histogram.
% figure(1), imhist(gray);

% Do linear transformation.
new = zeros(size(gray));
gray = im2double(gray);
% For all pixels, applying the linear transformation function.
for i = 1:size(gray,1)
    for j = 1:size(gray,2)
        new(i,j) = k + m * gray(i,j);
    end
end
new = im2uint8(new);
% Display the new image.
figure(2),imshow(new);
% figure(2),imhist(new);
title('New image');
end

