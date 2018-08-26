#coding:utf-8
import numpy as np
import cv2

# 载入图片以及初始化
img = cv2.imread('datas/f3.jpg')
# GrabCut所需内部调用的参数
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
iteration = 10
# 鼠标点击的时候变为True
drawing = False
# 如果为True, 选择前景。点击'm'或'1'或右键切换到背景选择
mode = True
# 记录鼠标点击时的位置
ix,iy = -1,-1
# 绘制点的半径
r = 8

# 将图片,以及选择的前景,背景一起显示出来
def merge(mask_fore, mask_back, img):
    mask = mask_fore[:,:,1:2]/255 + mask_back[:,:,2:3]/255
    if np.max(np.max(np.max(mask)))>1:
        return False, img
    mask = mask.astype('uint8')
    return True, mask_fore + mask_back + img*(1-mask)

# 鼠标回调函数
# 前景用绿色,背景用红色,不允许二者出现重叠!
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode,r
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    if event == cv2.EVENT_RBUTTONUP:
        mode = not mode

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing is True:
            if mode is True:
                cv2.circle(mask_fore,(x,y),r,(0,255,0),-1)
            else:
                cv2.circle(mask_back,(x,y),r,(0,0,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode is True:
            cv2.circle(mask_fore,(x,y),r,(0,255,0),-1)
        else:
            cv2.circle(mask_back,(x,y),r,(0,0,255),-1)
            
# 初始化前景和背景的mask
mask_fore = np.zeros_like(img)
mask_back = np.zeros_like(img)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while True:
    valid, masked = merge(mask_fore, mask_back, img)
    if not valid:
        print('Foreground has interaction with background! Please restart!')
        mask_fore = np.zeros_like(img)
        mask_back = np.zeros_like(img)
    cv2.imshow('image',masked)
    k = cv2.waitKey(1) & 0xFF
    
    # change mode
    if k == ord('m') or k == ord('1'):
        mode = not mode
        if mode:
            print('Drawing foreground')
        else:
            print('Drawing background')
    # clear all mask
    elif k== ord('r') or k == ord('2'):
        print('Restart')
        mask_fore = np.zeros_like(img)
        mask_back = np.zeros_like(img)
    # clear foreground mask
    elif k== ord('4'):
        print('Reselect foreground')
        mask_fore = np.zeros_like(img)
    # clear background mask
    elif k== ord('5'):
        print('Reselect background')
        mask_back = np.zeros_like(img)
    # run grabcut algorithm
    elif k== ord('c') or k == ord('3') or k == 13:
        print('Cutting foreground from the picture')
        mask_global = np.zeros(img.shape[:2],np.uint8) + 2
        mask_global[mask_fore[:,:,1] == 255] = 1
        mask_global[mask_back[:,:,2] == 255] = 0
        mask_global, bgdModel, fgdModel = cv2.grabCut(img,mask_global,None,bgdModel,fgdModel,iteration,cv2.GC_INIT_WITH_MASK)        
        mask_global = np.where((mask_global==2)|(mask_global==0),0,1).astype('uint8')
        target = img*mask_global[:,:,np.newaxis]
        cv2.imshow('target',target)
        cv2.imwrite('result.jpg',target)
    elif k == 27:
        break
cv2.destroyAllWindows()