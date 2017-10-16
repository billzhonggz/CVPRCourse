im = imread('0.jpg');
imgreen = im(:,:,2);
imshow(imgreen)
line([1,512],[256,256],'color','r')
plot(imgreen(256,:));