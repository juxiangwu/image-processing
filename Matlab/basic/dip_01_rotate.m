clc;
clear;
img = imread('../../datas/f4.jpg');
%Ðý×ª90¶È
dst = imrotate(img,90);
imshow(dst);