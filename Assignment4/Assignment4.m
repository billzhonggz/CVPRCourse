% Assignment 4
% Computer Vision and Pattern Recognition
% 1430003045 ÷”æ˚»Â
% Keep sizes of all images at 640*480
% Reference: https://cn.mathworks.com/matlabcentral/fileexchange/48479-face-recognition-using-eigenfaces

% Run training script.
[images,H,W,M,m,U,projection] = training('TrainingSet');

% Run test script.
testing('test1.jpg',images,H,W,M,m,U,projection);