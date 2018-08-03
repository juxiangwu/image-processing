clc;
clear;
img = imread('../../datas/f4.jpg');
cropped = imcrop(img);
imwrite(cropped,'../../temp/cropped.jpg','JPEG');
figure;
imshow(cropped);