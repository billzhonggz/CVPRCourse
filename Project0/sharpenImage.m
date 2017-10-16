function [ ] = sharpenImage( image,kernel )
%Sharpen image using convolution.
%   1430003045 ÖÓ¾ûÈå

% Transfer the image into gray.
gray = rgb2gray(image);
% Display the original image.
figure(1), imshow(gray);
%figure(1), imhist(gray);
title('Original image');

% Do cross-correlation.
new = conv2(gray,kernel);

% Display the new image.
figure(2), imshow(new);
%figure(2), imhist(new);
title('New image');
end

