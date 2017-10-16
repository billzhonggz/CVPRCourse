% Project 0, CVPR, Semster 1, 2017-2018
% 1430003045 ÷”æ˚»Â

% Importing and show an image.
im = imread('1.jpg');
imshow(im);

% Run histogram function.
histImg(im);

% Run histogram equalization.
histEqal(im);

% Run linear transformation.
linTransformImg(im,1,2);

% Set up kernel.
kernel = ones(3);

% Run smoothing.
smoothImg(im,kernel);

% Run sharpening.
sharpenImage(im,kernel);