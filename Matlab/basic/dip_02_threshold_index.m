clc;
clear;
img = imread('../../datas/f4.jpg');
subplot(1,3,1)
imshow(img)
title('Source Image');
threshold = 150;

img(img > threshold) = 255;
img(img <= threshold) = 0;

subplot(1,3,2)
imshow(img)
title('RGB Threshold');

% 转换成二值图像
img = logical(img);
subplot(1,3,3)
imshow(img)
title('Binary Image');