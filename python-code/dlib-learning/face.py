
import os
import dlib
import glob
import numpy as np
import cv2

print('初始化参数')
predictor_path = '1.dat'
face_rec_model_path = '2.dat'
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)


def get_descriptors(paths):
    '''
    计算指定路径下所有照片的特征
    '''
    print('计算所有候选人的人脸特征')
    descriptors = []
    for f in paths:
        print("Processing file: {}".format(f))
        img = cv2.imread(f)
        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        for k, d in enumerate(dets):  
            shape = sp(img, d)
            face_descriptor = facerec.compute_face_descriptor(img, shape)

            v = np.array(face_descriptor)  
            descriptors.append(v)
    return descriptors


def get_dist(img, descriptors, dets):
    '''
    计算目标照片与所有候选人照片的欧氏距离,返回一个列表
    '''
    dist = []
    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        d_test = np.array(face_descriptor) 
    
        # 计算欧式距离
        for i in descriptors:
            dist_ = np.linalg.norm(i-d_test)
            dist.append(dist_)
    return dist



def scan_images(faces_folder):
    print('正在扫描人脸库')
    paths = glob.glob(os.path.join(faces_folder, "*.jpg"))
    paths = sorted(paths)
    print('扫描完毕')
    names = []
    for each in paths:
        name = os.path.basename(each).split('.')[0]
        names.append(name)
    
    
    return  names, paths

def recognize_face(img, descriptors, candidates):
    '''
    开始人脸识别
    '''
    dets = detector(img, 1)
    dist = get_dist(img, descriptors, dets)
    if len(dist) == 0:
        return 0
    
    if min(dist) < 0.5 :
        c_d = dict(zip(candidates,dist))
        cd_sorted = sorted(c_d.items(), key=lambda d:d[1])

        #dlib.hit_enter_to_continue()
        return cd_sorted[0][0]
    else:
        
        return -1