function [] = histImg( image )
% Drawing histogram for an input image.
% 1430003045 ÷”æ˚»Â

% Transfer to gray.
gray = rgb2gray(image);
% Get the width and length of the image.
[M,N] = size(gray);

% Initialize vectors with the length of 256
t = 1:256;
n = 1:256;
count = 0;
% Looping in grey value.
for z = 1:256
    for i = 1:M
        for j = 1:N
            % Counting the number of matching current grey value.
            if gray(i,j) == z - 1
                count = count + 1;
            end
        end
    end
    % Record the count.
    t(z) = count;
    % Reset the counter.
    count = 0;
end
% Draw the histogram.
figure(1);
disp(t')
stem(n,t);
grid on;
title('Histogram of image');

% Draw the CDF.
counts = t;
x = n;
counts = counts/M/N;
figure(2);
plot(x,counts);
grid on;
title('CDF of the image');
end