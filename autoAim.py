'''Python implementation of an object detector based auto aim for video games'''
import time
import numpy as np
import cv2
from mss import mss
from simple_pid import PID
import torch
import win32con
import keyboard
import win32api
from autoAimUtils import getScreenInfo

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, verbose=False, )
model.to(DEVICE)
model.eval()

#User parameters
SHOW = True # boolean if the user wants to see the model output

CENTERING = .7 #centering is the decimal percentage of the screen to account for
THRESHOLD = .5 # threshold is the lowest confidence bound to be allowed to be considered a detection
WEAPONS_FREE = True # boolean should the program fire the weapon


targets = [0]
animals = [0, 15, 16, 17, 18, 19, 20, 21, 22, 23]

#set up PID for mouse control
Kp = 8
Ki = 0
Kd = 0

PID_LIMIT = 10000

#NOTE: this must be run before getScreenInfo, otherwise the reported values will be incorrect
sct = mss()

monitor = getScreenInfo(CENTERING)

x_center = monitor['width'] / 2
y_center = monitor['height'] / 2

# Begin the loop of screen inference
while True:

    startFrameTime = time.time() # begin a timer to calculate framerate

    img = np.array(sct.grab(monitor)) #capture the image

    predictions = model(img).pred # perform inference of the image.
    framerate = 1/(time.time() - startFrameTime) # stop the timer and calculate the framerate

    #add framerate to the cv2 image
    cv2.putText(img,
                str(framerate),
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA)

    # add the aim point to the image
    cv2.circle(img,
        (int(x_center), int(y_center)),
        radius=5,
        color=(0, 255, 0),
        thickness=5
    )


    if len(predictions) > 0: # if there were any predicted targets
        CLOSEST = 99999 # set an arbitrarily large value for distance to the center
        for box in predictions[0]: # loop through all predicted targets
            if box[4] > THRESHOLD and box[5] in targets: # if high confidence
                x = (box[2] + box[0]) / 2 # find the x pixel value for the target
                y = ((box[3] + box[1]) / 2) - ((box[3] - box[1]) / 3) # scale to aim for headshots
                class_num = int(box[5]) # find the class value for the target

                # add the target to the image
                cv2.circle(img,
                    (int(x), int(y)),
                    radius=10,
                    color=(255, 0, 0),
                    thickness=5
                )
                cv2.rectangle(img,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            color=(0, 255, 0),
                            thickness=2,
                            )
                # cv2.putText(img,
                #             org=(int(box[0]), int(box[1])),
                #             text=str(model.names[class_num]),
                #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #             fontScale=1,
                #             color=(255,0,0),
                #             thickness=2
                #             )

                # Calculate the distance to the target
                # negate the values to move the right direction
                dx = -int(x.item()-x_center)
                dy = -int(y.item()-y_center)
                dist = (dx**2 + dy**2)**(.5)

                # update the colosest target
                if dist < CLOSEST:
                    CLOSEST = dist

                    # add distance to the target
                    cv2.putText(img,
                                org=(int(box[0]), int(box[1])),
                                text=str(dx) + " " + str(dy),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1,
                                color=(255,0,0),
                                thickness=2
                                )

        # When the user requests, control the mouse
        if keyboard.is_pressed('f1') and CLOSEST < 99999:
            #check if continuation of autoAim from last frame

            if not AUTO_AIM:
                startTime = time.time() #get the start time
                AUTO_AIM=True
                #initiate the PID
                xpid = PID(Kp, Ki, Kd, setpoint=0)
                ypid = PID(Kp, Ki, Kd, setpoint=0)
                xpid.output_limits = (-PID_LIMIT, PID_LIMIT)
                ypid.output_limits = (-PID_LIMIT, PID_LIMIT)
            else:
                lastTime = time.time()
                dt = lastTime - startTime

                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, # move the mouse
                                        int(xpid(dx)*dt), # x increment to move
                                        int(ypid(dy)*dt), # y incrememnt to move
                                        )
                if dist < 50 and WEAPONS_FREE:
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
                    time.sleep(0.001)
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)
                startTime = lastTime #reset my timer
        else:
            AUTO_AIM=False

    if SHOW:
        cv2.imshow('Model View', img)

        # The wait insures the image is shown
        # the other portion of the command allows the user to stop the program by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows() # destroy the image window
            break # step out of the loop
