function [images,H,W,M,m,U,projection]=training(trainingFolder)
    % Identify directory of training set.
    trainingSet=dir(sprintf('%s/*.jpg',trainingFolder));
    % Read the first image to determine the Height & Width
    im=imread(fullfile(trainingFolder, trainingSet(1).name)); 
    % Get height, width and amount
    H=size(im,1);
    W=size(im,2);
    M=size(trainingSet,1); 
    % Initialize container.
    images=zeros(H,W,M);
    vec=zeros(H*W,M);
    % Load the training images
    for i=1:M
        images(:,:,i)=rgb2gray(imread(fullfile(trainingFolder, trainingSet(i).name)));
        % Do linear transformation
        vec(:,i)=reshape(images(:,:,i),H*W,1);
    end
    % Get the mean
    m=sum(vec,2)/M;
    % Face space
    faceSpace=vec-repmat(m,1,M);
    % Get an asymmetric matrix.
    L=faceSpace'*faceSpace;
    % Get eigenvector by Matlab stock function.
    [V,lambda]=eig(L);
    % Eigenvector of the covariance matrix of A. These are the eigenfaces
    U=faceSpace*V;
    % Projection of each vector in the face space A on the eigenfaces
    projection=U'*faceSpace;
end

