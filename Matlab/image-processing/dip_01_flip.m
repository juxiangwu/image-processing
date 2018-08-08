clc;
clear;
img = imread('../../datas/f4.jpg');
dstlr = fliplr(img);
dstup = flipud(img);
figure()
subplot(1,2,1);
imshow(dstlr);
subplot(1,2,2);
imshow(dstup);