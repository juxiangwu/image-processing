#utf-8
import numpy as np
import cv2
def filter_lomo(src):
    filter_map = cv2.imread('resources/images/lomoMap.png')
    filter_map = cv2.cvtColor(filter_map,cv2.COLOR_BGR2RGB)
    map_b = filter_map[2,:,2].copy().reshape((256,))
    map_g = filter_map[1,:,1].copy().reshape((256,))
    map_r = filter_map[0,:,0].copy().reshape((256,))
    dst = np.zeros_like(src)
    dst[:,:,0] = map_r[src[:,:,0]]
    dst[:,:,1] = map_r[src[:,:,1]]
    dst[:,:,2] = map_r[src[:,:,2]]
    return dst