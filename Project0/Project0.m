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
linTransformImg(im,0.5,1);

% Run smoothing.
smoothImg(im,10,5);

% Run sharpening.
sharpenImage(im,5);