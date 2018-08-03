clc;
clear;
img = imread('../../datas/f4.jpg');
subplot(1,2,1)
imshow(img)
title('Source Image');
threshold = 150;

for y = 1 : size(img,1)
    for x = 1 : size(img,2)
        
        if img(y,x,1) > threshold
            img(y,x,1) = 255;
        else
            img(y,x,1) = 0;
        end
        
         if img(y,x,2) > threshold
            img(y,x,2) = 255;
        else
            img(y,x,2) = 0;
         end
         
         if img(y,x,3) > threshold
            img(y,x,3) = 255;
        else
            img(y,x,3) = 0;
         end
    end
end

subplot(1,2,2)
imshow(img)
title('RGB Threshold');