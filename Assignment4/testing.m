function testing(testImage,images,H,W,M,m,U,projection)
    % Load the test image to be recognized
    testIm=imread(testImage);
    % Transform RGB to gray.
    testIm=rgb2gray(testIm)
    im=reshape(testIm,H*W,1);
    imtest=double(im);
    % Get distance of test image between mean.
    imd=imtest-m;
    % Projection of the test face on the eigenfaces
    om=U'*imd;
    d=repmat(om,1,M)-projection;
    % Initialize distance container.
    dist=zeros(M,1);
    % Find the distance from all training faces
    for i=1:M
        dist(i,1)=norm(d(:,i));
    end
    % Find the image index corresponding to the minimum of the distances
    index=IndexOfMinimum(dist);
    % Show the results
    subplot(1,2,1)
    imshow(testImage)
    title('Test face')
    subplot(1,2,2)
    imshow(uint8(images(:,:,index)))
    title('Recognized face')
end