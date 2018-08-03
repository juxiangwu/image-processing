#coding:utf-8

import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import cv2
import matplotlib.pyplot as plt
from filters.F1997 import filter_f1997
from filters.Lomo import filter_lomo
from filters.Amaro import filter_amaro
src = cv2.imread('resources/images/city.jpg')
src = cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
dst = filter_amaro(src)

plt.subplot(121)
plt.imshow(src)
plt.title('SRC')
plt.subplot(122)
plt.imshow(dst)
plt.title('Lomo')

plt.show()

# lomo_map = cv2.imread('resources/images/amaroMap.png')
# lomo_map = cv2.cvtColor(lomo_map,cv2.COLOR_BGR2RGB)
# print(lomo_map.shape)
# lomo_map_r = lomo_map[0,:,0]
# lomo_map_g = lomo_map[1,:,1]
# lomo_map_b = lomo_map[2,:,2]
# print(lomo_map_1_g)

# filter_map = cv2.imread('resources/images/amaroMap.png')
# filter_map = cv2.cvtColor(filter_map,cv2.COLOR_BGR2RGB)
# map_b = filter_map[2,:,2].copy().reshape((256,))
# map_g = filter_map[1,:,1].copy().reshape((256,))
# map_r = filter_map[0,:,0].copy().reshape((256,))
# print(map_g)