import cv2
import numpy as np
import serial
import time

Arduino = serial.Serial("COM3",9600)
time.sleep(2)


cam = cv2.VideoCapture(0)
lower_red = np.array([0,125,125])
upper_red = np.array([10,255,255])

while(True):
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)

    w= frame.shape[1]
    h = frame.shape[0]
    image_smooth = cv2.GaussianBlur(frame, (7,7), 0)

    mask = np.zeros_like(frame)

    mask[50:350, 50:350] = [255,255,255]

    image_roi = cv2.bitwise_and(image_smooth, mask)
    cv2.rectangle(frame, (50,50), (350,350), (0,0,255),2)
    cv2.line(frame, (150,50), (150,350), (0,0,255),1)
    cv2.line(frame, (250,50), (250,350), (0,0,255),1)

    cv2.line(frame, (50,150), (350,150), (0,0,255),1)
    cv2.line(frame, (50,250), (350,250), (0,0,255),1)



    
    
    image_hsv = cv2.cvtColor(image_roi, cv2.COLOR_BGR2HSV)
    image_threshold = cv2.inRange(image_hsv, lower_red, upper_red)
    contours, hierachy = cv2.findContours(image_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if(len(contours)!=0):
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]

        M = cv2.moments(cnt)
        if(M['m00']!=0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.circle(frame, (cx,cy), 4, (0, 255, 0), -1)

            if(cx in range(150,250)):
                if cy<150:
                    Arduino.write(b'f')
                    print("Forward")
                elif cy>250:
                    Arduino.write(b'b')
                    print("Backward")
                else:
                    Arduino.write(b's')
                    print("Stop")



            if(cy in range(150,250)):
                if cx<150:
                    Arduino.write(b'l')
                    print("Left")

                elif cx>250:
                    Arduino.write(b'r')
                    print("Right")

                else:
                    Arduino.write(b's')
                    print("Stop")
                    
                    

            
                
        x_bound, y_bound, w_bound, h_bound = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x_bound,y_bound), (x_bound+w_bound,y_bound+h_bound), (255,0,0),2)
    
    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1000)
    if key==ord("s"):
        break


cam.release()
cv2.destroyAllWindows()
