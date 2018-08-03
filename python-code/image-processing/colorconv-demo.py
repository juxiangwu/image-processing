#coding:utf-8
import sys
import os
curdir,fname = os.path.split(__file__)
sys.path.append(curdir)

from utils import colorconv
import matplotlib.pyplot as plt
import imageio
.
src = imageio.imread('datas/f4.jpg')

# xyz = colorconv.rgb2xyz(src)
# rgb = colorconv.xyz2rgb(xyz)

# plt.figure()
# plt.subplot(1,3,1)
# plt.imshow(src)
# plt.title('Source Image')
# plt.subplot(1,3,2)
# plt.imshow(xyz)
# plt.title('RGB->XYZ')
# plt.subplot(1,3,3)
# plt.imshow(rgb)
# plt.title('XYZ->RGB')

lab = colorconv.rgb2lab(src)
rgb = colorconv.lab2rgb(lab)
plt.figure()
plt.subplot(1,3,1)
plt.imshow(src)
plt.title('Source Image')
plt.subplot(1,3,2)
plt.imshow(colorconv.convertAbsScale(lab))
plt.title('RGB->LAB')
plt.subplot(1,3,3)
plt.imshow(colorconv.convertAbsScale(rgb))
plt.title('LAB->RGB')

plt.show()
