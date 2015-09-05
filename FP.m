% Face recognition by Santiago Serrano
clear all
close all
%clc

% number of images on your training set.
M = 20;

%Chosen std and mean. 
%It can be any number that it is close to the std and mean of most of the images.
cm = 100;
cstd = 80;

%read and show images(bmp);
figure(1);
for i=1:M
    st = 'F:\课\数学综合训练\Face perception\ORL_92x112\ORL_92x112\';
    if i<10
        st = strcat(st,'00');
    else
        st = strcat(st,'0');
    end
    str=strcat(st,int2str(i),'01.bmp');   %concatenates two strings that form the name of the image
    img=imread(str);
    subplot(ceil(sqrt(M)),ceil(sqrt(M)),i)
    imshow(img)
    if i==ceil(sqrt(M)/2)
        title('Training set','fontsize',18)
    end
    drawnow;
    [row, col] = size(img);    % get the number of rows (N1) and columns (N2)
    if i==1
        S = zeros(row*col,M);
    end
    temp = reshape(img',row*col,1);     %creates a (N1*N2)x1 matrix
    S(:,i) = temp;         %X is a N1*N2xM matrix after finishing the sequence
                        %this is our S
end


%Here we change the mean and std of all images. We normalize all images.
%This is done to reduce the error due to lighting conditions.
for i=1:size(S,2)
    temp = double(S(:,i));
    m = mean(temp);
    st = std(temp);
    S(:,i) = (temp-m)*cstd/st+cm;
end

%show normalized images
figure(2);
for i=1:M
    img=reshape(S(:,i),col,row);
    img=uint8(img');
    subplot(ceil(sqrt(M)),ceil(sqrt(M)),i)
    imshow(img)
    drawnow;
    if i==ceil(sqrt(M)/2)
        title('Normalized Training Set','fontsize',18)
    end
end


%mean image;
m = mean(S,2);   %obtains the mean of each row instead of each column
tmimg = uint8(m);   %converts to unsigned 8-bit integer. Values range from 0 to 255
img = reshape(tmimg,col,row);    %takes the N1*N2x1 vector and creates a N2xN1 matrix
img = img';       %creates a N1xN2 matrix by transposing the image.
figure(3);
imshow(img);
title('Mean Image','fontsize',18)

% Change image for manipulation
dbx = double(S);
L=dbx'*dbx;%Covariance matrix
% vv are the eigenvector for L
% dd are the eigenvalue for both L=dbx'*dbx and C=dbx*dbx';
[vv, dd]=eig(L);
% Sort and eliminate those whose eigenvalue is zero
[rr,cc] = size(vv);
v = zeros(rr,cc);
d = zeros(1,cc);
for i=1:size(vv,2)
    if(dd(i,i)>1e-4)
        v(:,i) = vv(:,i);
        d(i) = dd(i,i);
    end
 end
 
 %sort,  will return an ascending sequence
 [B, index]=sort(d);
 ind=zeros(size(index));
 dtemp=zeros(size(index));
 vtemp=zeros(size(v));
 len=length(index);
 for i=1:len
    dtemp(i)=B(len+1-i);
    ind(i)=len+1-index(i);
    vtemp(:,ind(i))=v(:,i);
 end
 d=dtemp;
 v=vtemp;

%Normalization of eigenvectors
 for i=1:size(v,2)       %access each column
   kk=v(:,i);
   temp=sqrt(sum(kk.^2));
   v(:,i)=v(:,i)./temp;
end

%Principal component
pc = dbx*v;
for i=1:size(v,2)
    temp=sqrt(d(i));
    pc(:,i) = pc(:,i)./temp;
end

%Normalization of eigenvectors
for i=1:size(pc,2)
    kk=pc(:,i);
    temp=sqrt(sum(kk.^2));
	pc(:,i)=pc(:,i)./temp;
end


% show eigenfaces;
figure(4);
for i=1:size(pc,2)
    img=reshape(pc(:,i),col,row);
    img=img';
    img=histeq(img,255);
    subplot(ceil(sqrt(M)),ceil(sqrt(M)),i)
    imshow(img)
    drawnow;
    if i==3
        title('Eigenfaces','fontsize',18)
    end
end

% Find the weight of each face in the training set.
omega = pc'*dbx;

% Acquire new image
% Note: the input image must have a bmp or jpg extension. 
%       It should have the same size as the ones in your training set. 
%       It should be placed on your desktop 

%InputImage = input('Please enter the name of the image and its extension \n','s');
InputImage = 'F:\课\数学综合训练\Eigen kit\03001.bmp';
InputImage = imread(strcat(InputImage));
InputImage = imresize(InputImage,[row col]);

figure(5)
subplot(1,2,1)
imshow(InputImage); colormap('gray');title('Input image','fontsize',18)
InImage = reshape(double(InputImage)',row*col,1);
temp = InImage;
me = mean(temp);
st = std(temp);
temp = (temp-me)*cstd/st+cm;

NormImage = temp;
Difference = temp-m;

p = pc'*NormImage;%计算每个主成分的权重
ReshapedImage = m + pc*p;    %m is the mean image, pc is the principal component
ReshapedImage = reshape(ReshapedImage,col,row);
ReshapedImage = ReshapedImage';
%show the reconstructed image.
subplot(1,2,2)
imagesc(ReshapedImage); colormap('gray');
title('Reconstructed image','fontsize',18)

InImWeight = pc'*Difference;
ll = 1:M;
figure(6)
subplot(1,2,1)
stem(ll,InImWeight)
title('Weight of Input Face','fontsize',14)

% Find Euclidean distance
cc = size(omega,2);
e = zeros(1,cc);
for i=1:cc
    q = omega(:,i);
    DiffWeight = InImWeight-q;
    mag = norm(DiffWeight);
    e(i) = mag;
end

kk = 1:size(e,2);
subplot(1,2,2)
stem(kk,e)
title('Eucledian distance of input image','fontsize',14)

MaximumValue=max(e)
MinimumValue=min(e)