import cv2
import numpy as np

# 3840 2160
cap = cv2.VideoCapture("data/02.mov")
feature_params = dict( maxCorners = 100,qualityLevel = 0.3,minDistance = 7,blockSize = 7 )
lk_params = dict(winSize  = (15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0,255,(100,3))
ret, old_frame = cap.read()
old_frame = cv2.resize(old_frame, (3840//4 , 2160//4))
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

while(1):
    ret,frame = cap.read()
    if frame is None: break
    frame = cv2.resize(frame, (3840//4 , 2160//4))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # hsv = np.zeros_like(frame)
    # hsv[...,1] = 255
    # mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1]) 
    # hsv[...,0] = ang*180/np.pi/2 
    # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX) 
    # bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    bgr = np.zeros_like(frame)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    val = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr[...,1] = val
    bgr[...,0] = val
    bgr[...,2] = val

    # step = 10
    # h, w = frame.shape[:2]
    # y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    # fx, fy = flow[y, x].T
    # lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    # lines = np.int32(lines)
    # line = []
    # for l in lines:
    #     if l[0][0]-l[1][0]>3 or l[0][1]-l[1][1]>3:
    #         line.append(l)
    # cv2.polylines(frame, line, 0, (0,255,255))
    # bgr = frame
    
    cv2.imshow('frame',bgr)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    old_gray = frame_gray.copy()

cv2.destroyAllWindows()
cap.release()

# frame_1 = cv2.imread("data/01.jpg")
# frame_2 = cv2.imread("data/02.jpg")
# frame_gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
# frame_gray_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

# flow = cv2.calcOpticalFlowFarneback(frame_gray_1, frame_gray_2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
# # # 绘制线
# # step = 10
# # h, w = frame_2.shape[:2]
# # y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
# # fx, fy = flow[y, x].T
# # lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
# # lines = np.int32(lines)

# # line = []
# # for l in lines:
# #     if l[0][0]-l[1][0]>3 or l[0][1]-l[1][1]>3:
# #         line.append(l)

# # cv2.polylines(frame_2, line, 0, (0,255,255))
# # cv2.imshow('flow', frame_2)
# # cv2.waitKey(0)
# print(flow)
# hsv = np.zeros_like(frame_1)
# hsv[...,1] = 255
# mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1]) 
# hsv[...,0] = ang*180/np.pi/2 
# hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX) 
# bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR) 
# cv2.imshow('flow', bgr)
# cv2.waitKey(0)