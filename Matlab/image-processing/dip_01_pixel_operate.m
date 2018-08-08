clc;
clear;
img = imread('../../datas/f4.jpg');

%for y = 1 : 100
  %  for x = 1 : 100
    %    img(y,x,1) = 255;
      %  img(y,x,2) = 0;
        %img(y,x,3) = 0;
    %end
%end
figure
subplot(2,2,1)
imshow(img);
img(1:250,1:100,1) = 0;
img(1:250,1:100,2) = 255;
img(1:250,1:100,3) = 0;
subplot(2,2,2)
imshow(img)