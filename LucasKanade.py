import numpy as np
import cv2

# 3840 2160
cap = cv2.VideoCapture("data/02.mov")
feature_params = dict( maxCorners = 100,qualityLevel = 0.3,minDistance = 7,blockSize = 7 )
lk_params = dict(winSize  = (15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0,255,(100,3))
ret, old_frame = cap.read()
old_frame = cv2.resize(old_frame, (3840//4 , 2160//4))
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    if frame is None: break
    frame = cv2.resize(frame, (3840//4 , 2160//4))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if(p1 is not None):
        good_new = p1[st==1]
        good_old = p0[st==1]

        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)
    else:
        img = frame

    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()

'''
frame_1 = cv2.imread("data/01.jpg")
frame_2 = cv2.imread("data/02.jpg")
frame_gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
frame_gray_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
feature_params = dict( maxCorners = 100,qualityLevel = 0.3,minDistance = 7,blockSize = 7 )
lk_params = dict(winSize  = (15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0,255,(100,3))

p0 = cv2.goodFeaturesToTrack(frame_gray_1, mask = None, **feature_params)#选取好的特征点，返回特征点列表
mask = np.zeros_like(frame_1)

p1, st, err = cv2.calcOpticalFlowPyrLK(frame_gray_1, frame_gray_2, p0, None, **lk_params)#计算新的一副图像中相应的特征点额位置
good_new = p1[st==1]
good_old = p0[st==1]

for i,(new,old) in enumerate(zip(good_new,good_old)):
    a,b = new.ravel() #ravel()函数用于降维并且不产生副本
    c,d = old.ravel()
    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    frame = cv2.circle(frame_2,(a,b),5,color[i].tolist(),-1)
img = cv2.add(frame,mask)

cv2.imshow('frame',img)
cv2.waitKey(0)
'''