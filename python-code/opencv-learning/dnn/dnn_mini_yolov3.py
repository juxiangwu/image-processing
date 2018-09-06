import os
from ctypes import *

class object(Structure):
    _fields_ = [("name", c_char_p),
                ("prob", c_float),
                ("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class objects(Structure):
    _fields_ = [("objs", POINTER(object)),
                ("cnt", c_int)]

if __name__ == "__main__":

    lib = cdll.LoadLibrary('darknet.dll')#CDLL("libYOLOv3SE.dll", RTLD_GLOBAL)
    load_net = lib.load_network
    load_net.argtypes = []
    load_net.restype = c_void_p

    detect = lib.network_predict
    detect.argtypes = [c_char_p,c_char_p]
    detect.restype = objects

    #load_net(b'data/yolov3-tiny.cfg',b'data/yolov3-tiny.weights') # tiny version
    load_net(b'../resources/models/yolo/yolov3-tiny.cfg',b'../resources/models/yolo/yolov3-tiny.weights')
    img='../resources/images/dog.jpg'
    out='temp/yolo-result.jpg'
    # for img in os.listdir('../darknet/data'):
    #     if img.endswith('jpg'):
    #         out = 'results/pred_'+img.split('.')[0]
    #         img = '../darknet/data/'+img

    #         objs = detect(bytes(img,encoding='utf8'),bytes(out,encoding='utf8'))
            
    #         print("predition:")
    #         for i in range(objs.cnt):
    #             x = round(objs.objs[i].x * 100,2)
    #             y = round(objs.objs[i].y * 100,2)
    #             w = round(objs.objs[i].w * 100,2)
    #             h = round(objs.objs[i].h * 100,2)
    #             n = objs.objs[i].name
    #             prob = round(objs.objs[i].prob ,2)
    #             print('name: ',n,'prob: ',prob, '\nbox: ',x,y,w,h,'\n')
    #         print()
    objs = detect(bytes(img,encoding='utf8'),bytes(out,encoding='utf8'))
    print("predition:")
    for i in range(objs.cnt):
        x = round(objs.objs[i].x * 100,2)
        y = round(objs.objs[i].y * 100,2)
        w = round(objs.objs[i].w * 100,2)
        h = round(objs.objs[i].h * 100,2)
        n = objs.objs[i].name
        prob = round(objs.objs[i].prob ,2)
        print('name: ',n,'prob: ',prob, '\nbox: ',x,y,w,h,'\n')