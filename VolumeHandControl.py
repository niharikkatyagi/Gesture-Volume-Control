import cv2
import time
import numpy as np
import handtrackingmodule as htm
import math
import osascript

wCam, hCam = 640, 480
cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)


minVol = 0
maxVol = 100
volBar = 0
volPer = 0
while True:
    success, img = cap.read()
    img1 = detector.findHands(img)
    lmList = detector.findPosition(img1, draw=False)
    if len(lmList) != 0:
        #print(lmList[4],lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img1, (x1,y1), 10, (255,0,255), cv2.FILLED)
        cv2.circle(img1, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img1, (x1,y1), (x2,y2), (255,0,255), 3)
        cv2.circle(img1, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        #print(length)

        vol1 = np.interp(length, [30, 200], [minVol, maxVol])
        volBar = np.interp(length, [30, 200], [400, 150])
        volPer = np.interp(length, [30, 200], [0, 100])
        vol = "set volume output volume " + str(vol1)
        osascript.osascript(vol)
        print(vol1)

        if length<30:
            cv2.circle(img1, (cx, cy), 7, (0, 255, 0), cv2.FILLED)

        if length>200:
            cv2.circle(img1, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img1, (x2, y2), 10, (0, 255, 0), cv2.FILLED)
            cv2.line(img1, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(img1, (cx, cy), 7, (0, 255, 0), cv2.FILLED)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.rectangle(img1, (50, 150), (85,400), (0,255,0), 3)
    cv2.rectangle(img1, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img1, f"{int(volPer)} %", (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.putText(img1,f"FPS: {int(fps)}", (40,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)

    cv2.imshow("img", img1)
    cv2.waitKey(1)

