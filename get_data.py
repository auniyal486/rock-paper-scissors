import cv2 as cv
import os
labels = ["rock", "paper", "scissors", "background"]
save_path = "C:/Users/abhis/data/machine learning/rock_paper_scissors"
try:
    os.mkdir(save_path)
except FileExistsError:
    pass
for i in labels:
    camera = cv.VideoCapture(0)
    folders = os.path.join(save_path, i)
    try:
        os.mkdir(folders)
    except FileExistsError:
        pass
    start = False
    count = 0
    while count < 500:
        status,frame=camera.read()
        if not status:
            print("frame is not available")
            break
        cv.rectangle(frame,(50,50),(300,300),(255,255,255),2)
        if start:
            frame_part=frame[50:300,50:300]
            frame_part=cv.cvtColor(frame_part,cv.COLOR_BGR2GRAY)
            folder_img=os.path.join(folders,"{}img{}.jpg".format(i,count))
            cv.imwrite(folder_img,frame_part)
            count+=1
            cv.putText(frame,"{} collected {}".format(i,count),(50,40),cv.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,0,255),2)
        cv.imshow("webcame", frame)
        k=cv.waitKey(50)&0xFF
        if k==ord("s"):
            start=True
        if k==ord("q"):
            break
    camera.release()
    cv.destroyAllWindows()