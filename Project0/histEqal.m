function [ ] = histEqal( image )
%Histogram equalization
%   1430003045 ÷”æ˚»Â

% Transfer to gray.
gray = rgb2gray(image);
% Get the size of the image.
[M,N] = size(gray);
pixelCount = size(gray,1) * size(gray,2);
% Show the original gray image.
figure,imshow(gray);
title('Original image');
% Get the histogram of the orignal image.
hist = zeros(size(gray));
freq = zeros(256,1);
probf = zeros(256,1);

for i = 1:M
    for j = 1:N
        value = gray(i,j);
        freq(value + 1) = freq(value + 1) + 1;
        probf(value + 1) = freq(value + 1) / pixelCount;
    end
end

% Get the cumulative distribution.
sum = 0;
probc = zeros(256,1);
cum = zeros(256,1);
output = zeros(256,1);
for i = 1:size(probf)
    sum = sum + freq(i);
    cum(i) = sum;
    probc(i) = cum(i) / pixelCount;
    output(i) = round(probc(i) * 255);
end
for i = 1:M
    for j = 1:N
        hist(i,j) = output(gray(i,j) + 1);
    end
end
% Show the image.
figure, imshow(hist);
title('Histogram equalization');