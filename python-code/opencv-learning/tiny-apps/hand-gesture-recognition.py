#coding:utf-8
import cv2
import numpy as np
import math

class Gesture(object):
    def __init__(self,name):
        self.name=name
    def getName(self):
        return self.name
    def set_palm(self,hand_center,hand_radius):
        self.hand_center=hand_center
        self.hand_radius=hand_radius
    def set_finger_pos(self,finger_pos):
        self.finger_pos=finger_pos
        self.finger_count=len(finger_pos)
    def calc_angles(self):
        self.angle=np.zeros(self.finger_count,dtype=int)
        for i in range(self.finger_count):
            y = self.finger_pos[i][1]
            x = self.finger_pos[i][0]
            self.angle[i]=abs(math.atan2((self.hand_center[1]-y),(x-self.hand_center[0]))*180/math.pi)

def DefineGestures():
    dict={}
    # 1. BEGIN ------------------------------------#
    V=Gesture("V")
    V.set_palm((475,225),45)
    V.set_finger_pos([(490,90),(415,105)])
    V.calc_angles()
    dict[V.getName()]=V
    # 1. END --------------------------------------#
    # 2. BEGIN ------------------------------------#
    L_right=Gesture("L_right")
    L_right.set_palm((475,225),50)
    L_right.set_finger_pos([(450,62),(345,200)])
    L_right.calc_angles()
    dict[L_right.getName()]=L_right
    # 2. END --------------------------------------#
    # 3. BEGIN ------------------------------------#
    Index_Pointing=Gesture("Index_Pointing")
    Index_Pointing.set_palm((480,230),43)
    Index_Pointing.set_finger_pos([(475,102)])
    Index_Pointing.calc_angles()
    dict[Index_Pointing.getName()]=Index_Pointing
    # 3. END --------------------------------------#
    return dict

def CompareGestures(src1,src2):
    if(src1.finger_count==src2.finger_count):
        if(src1.finger_count==1):
            angle_diff=src1.angle[0]-src2.angle[0]
            if(angle_diff>20):
                result=0
            else:
                len1 = np.sqrt((src1.finger_pos[0][0]- src1.hand_center[0])**2 + (src1.finger_pos[0][1] - src1.hand_center[1])**2)
                len2 = np.sqrt((src2.finger_pos[0][0]- src2.hand_center[0])**2 + (src2.finger_pos[0][1] - src2.hand_center[1])**2)
                length_diff=len1/len2
                radius_diff=src1.hand_radius/src2.hand_radius
                length_score=abs(length_diff-radius_diff)
                if(length_score<0.09):
                    result=src2.getName()
                else:
                    result=0
        else:
            angle_diff=[]
            for i in range(src1.finger_count):
                angle_diff.append(src1.angle[i]-src2.angle[i])
            angle_score=max(angle_diff)-min(angle_diff)
            if(angle_score<15):
                length_diff=[]
                for i in range(src1.finger_count):
                    len1 = np.sqrt((src1.finger_pos[i][0]- src1.hand_center[0])**2 + (src1.finger_pos[i][1] - src1.hand_center[1])**2)
                    len2 = np.sqrt((src2.finger_pos[i][0]- src2.hand_center[0])**2 + (src2.finger_pos[i][1] - src2.hand_center[1])**2)
                    length_diff.append(len1/len2)
                length_score=max(length_diff)-min(length_diff)
                if(length_score<0.06):
                    result=src2.getName()
                else:
                    result=0
            else:
                result=0
    else:
        result=0
    return result

def DecideGesture(src,GestureDictionary):
    result_list=[]
    for k in GestureDictionary.keys():
        src2='"'+k+'"'
        result=CompareGestures(src,GestureDictionary[k])
        if(result!=0):
            return result
    return "NONE"


    # Variables & parameters
hsv_thresh_lower=150
gaussian_ksize=11
gaussian_sigma=0
morph_elem_size=13
median_ksize=3
capture_box_count=9
capture_box_dim=20
capture_box_sep_x=8
capture_box_sep_y=18
capture_pos_x=500
capture_pos_y=150
cap_region_x_begin=0.5 # start point/total width
cap_region_y_end=0.8 # start point/total width
finger_thresh_l=2.0
finger_thresh_u=3.8
radius_thresh=0.04 # factor of width of full frame
first_iteration=True
finger_ct_history=[0,0]

# ------------------------ Function declarations ------------------------ #

# 1. Hand capture histogram
def hand_capture(frame_in,box_x,box_y):
    hsv = cv2.cvtColor(frame_in, cv2.COLOR_BGR2HSV)
    ROI = np.zeros([capture_box_dim*capture_box_count,capture_box_dim,3], dtype=hsv.dtype)
    for i in xrange(capture_box_count):
        ROI[i*capture_box_dim:i*capture_box_dim+capture_box_dim,0:capture_box_dim] = hsv[box_y[i]:box_y[i]+capture_box_dim,box_x[i]:box_x[i]+capture_box_dim]
    hand_hist = cv2.calcHist([ROI],[0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hand_hist,hand_hist, 0, 255, cv2.NORM_MINMAX)
    return hand_hist

# 2. Filters and threshold
def hand_threshold(frame_in,hand_hist):
    frame_in=cv2.medianBlur(frame_in,3)
    hsv=cv2.cvtColor(frame_in,cv2.COLOR_BGR2HSV)
    hsv[0:int(cap_region_y_end*hsv.shape[0]),0:int(cap_region_x_begin*hsv.shape[1])]=0 # Right half screen only
    hsv[int(cap_region_y_end*hsv.shape[0]):hsv.shape[0],0:hsv.shape[1]]=0
    back_projection = cv2.calcBackProject([hsv], [0,1],hand_hist, [00,180,0,256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_elem_size,morph_elem_size))
    cv2.filter2D(back_projection, -1, disc, back_projection)
    back_projection=cv2.GaussianBlur(back_projection,(gaussian_ksize,gaussian_ksize), gaussian_sigma)
    back_projection=cv2.medianBlur(back_projection,median_ksize)
    ret, thresh = cv2.threshold(back_projection, hsv_thresh_lower, 255, 0)
    
    return thresh

# 3. Find hand contour
def hand_contour_find(contours):
    max_area=0
    largest_contour=-1
    for i in range(len(contours)):
        cont=contours[i]
        area=cv2.contourArea(cont)
        if(area>max_area):
            max_area=area
            largest_contour=i
    if(largest_contour==-1):
        return False,0
    else:
        h_contour=contours[largest_contour]
        return True,h_contour

# 4. Detect & mark fingers
def mark_fingers(frame_in,hull,pt,radius):
    global first_iteration
    global finger_ct_history
    finger=[(hull[0][0][0],hull[0][0][1])]
    j=0

    cx = pt[0]
    cy = pt[1]
    
    for i in range(len(hull)):
        dist = np.sqrt((hull[-i][0][0] - hull[-i+1][0][0])**2 + (hull[-i][0][1] - hull[-i+1][0][1])**2)
        if (dist>18):
            if(j==0):
                finger=[(hull[-i][0][0],hull[-i][0][1])]
            else:
                finger.append((hull[-i][0][0],hull[-i][0][1]))
            j=j+1
    
    temp_len=len(finger)
    i=0
    while(i<temp_len):
        dist = np.sqrt( (finger[i][0]- cx)**2 + (finger[i][1] - cy)**2)
        if(dist<finger_thresh_l*radius or dist>finger_thresh_u*radius or finger[i][1]>cy+radius):
            finger.remove((finger[i][0],finger[i][1]))
            temp_len=temp_len-1
        else:
            i=i+1        
    
    temp_len=len(finger)
    if(temp_len>5):
        for i in range(1,temp_len+1-5):
            finger.remove((finger[temp_len-i][0],finger[temp_len-i][1]))
    
    palm=[(cx,cy),radius]

    if(first_iteration):
        finger_ct_history[0]=finger_ct_history[1]=len(finger)
        first_iteration=False
    else:
        finger_ct_history[0]=0.34*(finger_ct_history[0]+finger_ct_history[1]+len(finger))

    if((finger_ct_history[0]-int(finger_ct_history[0]))>0.8):
        finger_count=int(finger_ct_history[0])+1
    else:
        finger_count=int(finger_ct_history[0])

    finger_ct_history[1]=len(finger)

    count_text="FINGERS:"+str(finger_count)
    cv2.putText(frame_in,count_text,(int(0.62*frame_in.shape[1]),int(0.88*frame_in.shape[0])),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,255),1,8)

    for k in range(len(finger)):
        cv2.circle(frame_in,finger[k],10,255,2)
        cv2.line(frame_in,finger[k],(cx,cy),255,2)
    return frame_in,finger,palm

# 5. Mark hand center circle

def mark_hand_center(frame_in,cont):    
    max_d=0
    pt=(0,0)
    x,y,w,h = cv2.boundingRect(cont)
    for ind_y in xrange(int(y+0.3*h),int(y+0.8*h)): #around 0.25 to 0.6 region of height (Faster calculation with ok results)
        for ind_x in xrange(int(x+0.3*w),int(x+0.6*w)): #around 0.3 to 0.6 region of width (Faster calculation with ok results)
            dist= cv2.pointPolygonTest(cont,(ind_x,ind_y),True)
            if(dist>max_d):
                max_d=dist
                pt=(ind_x,ind_y)
    if(max_d>radius_thresh*frame_in.shape[1]):
        thresh_score=True
        cv2.circle(frame_in,pt,int(max_d),(255,0,0),2)
    else:
        thresh_score=False
    return frame_in,pt,max_d,thresh_score

# 6. Find and display gesture

def find_gesture(frame_in,finger,palm):
    frame_gesture.set_palm(palm[0],palm[1])
    frame_gesture.set_finger_pos(finger)
    frame_gesture.calc_angles()
    gesture_found=DecideGesture(frame_gesture,GestureDictionary)
    gesture_text="GESTURE:"+str(gesture_found)
    cv2.putText(frame_in,gesture_text,(int(0.56*frame_in.shape[1]),int(0.97*frame_in.shape[0])),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,255),1,8)
    return frame_in,gesture_found

# 7. Remove bg from image

def remove_bg(frame):
    fg_mask=bg_model.apply(frame)
    kernel = np.ones((3,3),np.uint8)
    fg_mask=cv2.erode(fg_mask,kernel,iterations = 1)
    frame=cv2.bitwise_and(frame,frame,mask=fg_mask)
    return frame

# ------------------------ BEGIN ------------------------ #

# Camera
camera = cv2.VideoCapture(0)
capture_done=0
bg_captured=0
GestureDictionary=DefineGestures()
frame_gesture=Gesture("frame_gesture")

while(1):
    # Capture frame from camera
    ret, frame = camera.read()
    frame=cv2.bilateralFilter(frame,5,50,100)
    # Operations on the frame
    frame=cv2.flip(frame,1)
    cv2.rectangle(frame,(int(cap_region_x_begin*frame.shape[1]),0),(frame.shape[1],int(cap_region_y_end*frame.shape[0])),(255,0,0),1)
    frame_original=np.copy(frame)
    if(bg_captured):
        fg_frame=remove_bg(frame)
        
    if (not (capture_done and bg_captured)):
        if(not bg_captured):
            cv2.putText(frame,"Remove hand from the frame and press 'b' to capture background",(int(0.05*frame.shape[1]),int(0.97*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1,8)
        else:
            cv2.putText(frame,"Place hand inside boxes and press 'c' to capture hand histogram",(int(0.08*frame.shape[1]),int(0.97*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1,8)
        first_iteration=True
        finger_ct_history=[0,0]
        box_pos_x=np.array([capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x],dtype=int)
        box_pos_y=np.array([capture_pos_y,capture_pos_y,capture_pos_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y],dtype=int)
        for i in range(capture_box_count):
            cv2.rectangle(frame,(box_pos_x[i],box_pos_y[i]),(box_pos_x[i]+capture_box_dim,box_pos_y[i]+capture_box_dim),(255,0,0),1)
    else:
        frame=hand_threshold(fg_frame,hand_histogram)
        contour_frame=np.copy(frame)
        contours,hierarchy=cv2.findContours(contour_frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        found,hand_contour=hand_contour_find(contours)
        if(found):
            hand_convex_hull=cv2.convexHull(hand_contour)
            frame,hand_center,hand_radius,hand_size_score=mark_hand_center(frame_original,hand_contour)
            if(hand_size_score):
                frame,finger,palm=mark_fingers(frame,hand_convex_hull,hand_center,hand_radius)
                frame,gesture_found=find_gesture(frame,finger,palm)
        else:
            frame=frame_original

    # Display frame in a window
    cv2.imshow('Hand Gesture Recognition v1.0',frame)
    interrupt=cv2.waitKey(10)
    
    # Quit by pressing 'q'
    if  interrupt & 0xFF == ord('q'):
        break
    # Capture hand by pressing 'c'
    elif interrupt & 0xFF == ord('c'):
        if(bg_captured):
            capture_done=1
            hand_histogram=hand_capture(frame_original,box_pos_x,box_pos_y)
    # Capture background by pressing 'b'
    elif interrupt & 0xFF == ord('b'):
        bg_model = cv2.BackgroundSubtractorMOG2(0,10)
        bg_captured=1
    # Reset captured hand by pressing 'r'
    elif interrupt & 0xFF == ord('r'):
        capture_done=0
        bg_captured=0
        
# Release camera & end program
camera.release()
cv2.destroyAllWindows()