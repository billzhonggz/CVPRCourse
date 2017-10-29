function [ fiE ] = edgDirct( im )
%Calculating edge direction.
%   1430003045 �Ӿ���

% Run direction of gradient function.
fiG = dirGrad(im);
% Get edge direction by add pi/2.
fiE = fiG + pi/2;

end

