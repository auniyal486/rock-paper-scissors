import cv2 as cv
import numpy as np
from keras.preprocessing import image
from  keras.models import model_from_json

with open('model.json', 'r') as f:
    loaded_model_json = f.read()
loaded_model =model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "rock":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "Computer"

    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissors":
            return "Computer"

    if move1 == "scissors":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"

camera=cv.VideoCapture(0)
start=False
while True:
    sucess,frame=camera.read()
    if not sucess:
        continue
    cv.rectangle(frame,(50,50),(300,300),(255,255,255),2)
    k=cv.waitKey(100)&0xFF    
    if(k==ord("s")or start):
        frameWithoutText=frame.copy()
        picture=frameWithoutText[50:300,50:300]
        picture=cv.cvtColor(picture,cv.COLOR_BGR2GRAY)
        picture=cv.resize(picture,(150,150),interpolation=cv.INTER_AREA)
        picture = image.img_to_array(picture)
        picture = np.expand_dims(picture, axis = 0)
        result=loaded_model.predict(picture)
        result_index=np.argmax(result[0])
        move=""
        if(result_index==0):
            move="none"
        elif(result_index==1):
            move="paper"
        elif(result_index==2):
            move="rock"
        else:
            move="scissors"
        if(move!="none"):
            cv.putText(frameWithoutText,"Your's Move : "+move,(320,50),cv.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,0,255),2)
            computer_move_name = np.random.choice(['rock', 'paper', 'scissors'])
            cv.putText(frameWithoutText,"Computer's Move : "+ computer_move_name,(50,450),cv.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,0,255),2)
            winner=calculate_winner(move, computer_move_name)
            cv.putText(frameWithoutText,"winner : "+ winner,(350,150),cv.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,0,255),2)
        else:
            cv.putText(frameWithoutText,"waiting of posture",(350,50),cv.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,0,255),2)
        cv.imshow("out",frameWithoutText)
        if(k==ord("b")):
            start=False
        else:
            start=True;
    else:
        frameWithText=frame.copy()
        cv.putText(frameWithText,"press 's' to start",(50,50),cv.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,0,255),2)
        cv.imshow("out",frameWithText)
    if k==ord("q"):
        break
camera.release()
cv.destroyAllWindows()
