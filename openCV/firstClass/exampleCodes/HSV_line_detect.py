import cv2
import numpy as np
import serial
import pyzbar.pyzbar as pyzbar


ser1 = serial.Serial("COM16", 9600, timeout=0)#차량
cap1 = cv2.VideoCapture('http://192.168.43.1:8080/video')
cv2.namedWindow('image')

ENDMARK = 'e'
color_list = [[[101, 124, 43], [131, 255, 211]],# 기본
              [[0, 127, 0], [30, 255, 242]],# 좌회전
              [[42, 65, 103], [69, 223, 233]],# 우회전
              ]

next_loc = ''
destination = ''

mode = 0 # 0은 방황, 1은 목적지가 있는 상태
k=0
while 1:
    text = ''
    count = 0
    sum = 0.0
    ret, frame = cap1.read()
    if ret == False:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    decoded = pyzbar.decode(gray)
    for d in decoded:
        text = d.data.decode("utf-8")
        if text == 'A': # 교차로에 왔을 때
            ser1.write('C'.encode())
            ser1.write(ENDMARK.encode())
            k = 1
        if text in "HI": # 목적지에 도착했을 때
            ser1.write('D'.encode())
            ser1.write(ENDMARK.encode())

    blur_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)
    low_c = np.array(color_list[k][0])
    up_c = np.array(color_list[k][1])
    mask = cv2.inRange(hsv, low_c, up_c)  # 적용
    edges = cv2.Canny(mask, 75, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50)
    if lines is not None:  # 이 안은 에지 선긋기 관련
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if (y2 - y1) != 0:
                angle = (x2 - x1) / (y2 - y1)
                if -10 < angle < 10:
                    sum += angle
                    count += 1
            cv2.line(frame, (x1, y1), (x2, y2), (51, 204, 255), 5)
        if count != 0:
            angle = sum / count
            angle *= 100
            if angle > 100:
                angle = 99
            if angle < -100:
                angle = -99
            angle = str(int(angle))
            ser1.write(angle.encode())
            ser1.write(ENDMARK.encode())


    cv2.imshow('image', frame)
    if cv2.waitKey(1) == ord('q'):
        break


cv2.destroyAllWindows()
