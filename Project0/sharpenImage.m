function [ ] = sharpenImage( image,variance )
%Sharpen image
%   1430003045 ÖÓ¾ûÈå

% Transfer the image into gray.
gray = rgb2gray(image);
% Display the original image.
figure(1), imshow(gray);
%figure(1), imhist(gray);
title('Original image');

% Initialize the custom filter.
H = padarray(2,[2,2]) - fspecial('gaussian',5,variance);
new = imfilter(gray,H);

% Display the new image.
figure(2), imshow(new);
%figure(2), imhist(new);
title('New image');
end

