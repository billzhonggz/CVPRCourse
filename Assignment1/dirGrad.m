function [ fiG ] = dirGrad( im )
%Calculate the direction of gradient.
%   1430003045 ÷”æ˚»Â

% Initialize required variables.
% Initialize the Sobel Operators.
Di = 1/32 * [3,0,-3;10,0,-10;3,0,-3];
Dj = 1/32 * [3,10,3;0,0,0;-3,-10,-3];
im = double(im);
gi = conv2(im,Di,'same');
gj = conv2(im,Dj,'same');
% Compute necessary variables.
gausFilter = fspecial('gaussian',[3,3],1.6);
H12 = conv2(gi.*gj,gausFilter,'same');
H11 = conv2(power(gi,2),gausFilter,'same');
H22 = conv2(power(gj,2),gausFilter,'same');
% Compute direction of gradient.
fiG = 1/2 * atan2d(2*H12,H11-H22);
end

