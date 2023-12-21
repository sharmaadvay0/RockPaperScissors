import cv2
import time
import os

videoCapture = cv2.VideoCapture(0)

rockStart = False
paperStart = False
scissorStart = False
rockImages = 0
paperImages = 0
scissorImages = 0
totalImages = 250

while True:
    ret, frame = videoCapture.read()
    frameWithRect = cv2.rectangle(frame, (290, 205), (610,525), (0,0,0), 5)
    cv2.imshow('Input', frame)

    if rockStart and rockImages < totalImages:
        cv2.imwrite(f"images/rock/{rockImages}.jpg", frame[215:515, 300:600])
        print("Collected Rock image " + str(rockImages))
        rockImages += 1
    
    if paperStart and paperImages < totalImages:
        cv2.imwrite(f"images/paper/{paperImages}.jpg", frame[215:515, 300:600])
        print("Collected Paper image " + str(paperImages))
        paperImages += 1

    if scissorStart and scissorImages < totalImages:
        cv2.imwrite(f"images/scissors/{scissorImages}.jpg", frame[215:515, 300:600])
        print("Collected Scissor image " + str(scissorImages))
        scissorImages += 1
    c = cv2.waitKey(1)

    if c == ord('r'):
        rockStart = True
        time.sleep(3)

    if c == ord('p'):
        paperStart = True
        time.sleep(3)
    
    if c == ord('s'):
        scissorStart = True
        time.sleep(3)

    if c == ord('1'):
        files = os.listdir("images/rock/")
        for f in files:
            filePath = os.path.join("images/rock/", f)
            if os.path.isfile(filePath):
                os.remove(filePath)
        rockImages = 0
        rockStart = False
        print("Deleted Rock Files")
    
    if c == ord('2'):
        files = os.listdir("images/paper/")
        for f in files:
            filePath = os.path.join("images/paper/", f)
            if os.path.isfile(filePath):
                os.remove(filePath)
        paperImages = 0
        paperStart = False
        print("Deleted paper Files")
    
    if c == ord('3'):
        files = os.listdir("images/scissors/")
        for f in files:
            filePath = os.path.join("images/scissors/", f)
            if os.path.isfile(filePath):
                os.remove(filePath)
        scissorImages = 0
        scissorStart = False
        print("Deleted scissors Files")

    if c == ord(' '):
        break


videoCapture.release()
cv2.destroyAllWindows()