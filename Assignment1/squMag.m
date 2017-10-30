function [ b ] = squMag( im )
%Custom function for Squared magnitude.
%   1430003045 ÷”æ˚»Â

% Initialize the Sobel Operators.
Di = 1/32 * [3,0,-3;10,0,-10;3,0,-3];
Dj = 1/32 * [3,10,3;0,0,0;-3,-10,-3];
% Compute the squared gradient.
im = double(im);
gi = conv2(im,Di,'same');
gj = conv2(im,Dj,'same');
% Compute the squre magnitude.
gausFilter = fspecial('gaussian',[3,3],1.6);
bi = conv2(power(gi,2),gausFilter,'same');
bj = conv2(power(gj,2),gausFilter','same');
% Export the squre magnitude.
b = bi + bj;
end