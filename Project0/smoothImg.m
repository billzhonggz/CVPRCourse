function [ ] = smoothImg( image,kernel,sigma )
%Smoothing image with averaging filter.
%   1430003045 ÖÓ¾ûÈå

% Transfer the image into gray.
gray = rgb2gray(image);
% Display the original image.
figure(1), imshow(gray);
%figure(3), imhist(gray);
title('Original image');

% Do smoothing
h = fspecial('gaussian', kernel, sigma);
mesh(h);
imagesc(h);
new = imfilter(gray,h);

% Display the new image.
figure(4), imshow(new);
%figure(5), imhist(new);
title('New image');
end

