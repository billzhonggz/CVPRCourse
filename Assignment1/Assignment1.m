% Assignment 1, Computer Vision and Pattern Recoginition
% 1430003045 ÷”æ˚»Â

% Load the image and transfer it to B&W.
im = imread('1.jpg');
im = rgb2gray(im);
figure(1);
imshow(im);
title('Original Image');

% Use MATLAB functions to show the edges.
sobelEdges = edge(im,'Sobel');
cannyEdges = edge(im,'Canny');
figure(2);
imshowpair(sobelEdges,cannyEdges,'montage');
title('Sobel / Canny edeges by MATLAB function');