#! /usr/bin/env python3

import copy
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from phue import Bridge
from soco import SoCo
import pygame
import time
import threading
import sys
# import tkinter
# import tkinker.ttk
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
# from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


# General Settings
prediction = ''
action = ''
score = 0
img_counter = 500


# pygame.event.wait()

class Volume(object):
    def __init__(self):
        self.level = .5

    def increase(self, amount):
        self.level += amount
        print(f'New level is: {self.level}')

    def decrease(self, amount):
        self.level -= amount
        print(f'New level is: {self.level}')


vol = Volume()

# Turn on/off the ability to save images, or control Philips Hue/Sonos
save_images, selected_gesture = False, 'peace'
smart_home = True

# Philips Hue Settings
bridge_ip = '192.168.0.95'
print('press the bridge button.')
# bridge_ip = input("your bridge ip: ")
b = Bridge(bridge_ip)
brightness = 254
on_command = {'transitiontime': 0, 'on': True, 'bri': brightness}
off_command = {'transitiontime': 0, 'on': False}

b.connect()

# Sonos Settings(deleted)
# sonos_ip = '192.168.0.6'
# sonos = SoCo(sonos_ip)

gesture_names = {0: 'Fist',
                 1: 'L',
                 2: 'Okay',
                 3: 'Palm',
                 4: 'Peace'}

model = load_model('./VGG_cross_validated.h5')


def predict_rgb_image(img):
    result = gesture_names[model.predict_classes(img)[0]]
    print(result)
    return (result)


def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    print(f'pred_array: {pred_array}')
    result = gesture_names[np.argmax(pred_array)]
    print(f'Result: {result}')
    print(max(pred_array[0]))
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(result)
    return result, score


# parameters
cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
threshold = 60  # binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0
bgModel = None

# variableslt
isBgCaptured = 0  # bool, whether the background captured
triggerSwitch = False  # if true, keyboard simulator works


def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


# Camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(10, 200)


def run1():
    isalreadyfist = False;
    while camera.isOpened():
        ret, frame = camera.read()
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        frame = cv2.flip(frame, 1)  # flip the frame horizontally
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                      (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

        cv2.imshow('original', frame)

        # Run once background is captured
        global isBgCaptured, prediction, score, action, smart_home, brightness
        if isBgCaptured == 1:
            img = remove_background(frame)
            img = img[0:int(cap_region_y_end * frame.shape[0]),
                  int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
            # cv2.imshow('mask', img)

            # convert the image into binary image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
            # cv2.imshow('blur', blur)
            ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Add prediction and action text to thresholded image
            # cv2.putText(thresh, f"Prediction: {prediction} ({score}%)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
            # cv2.putText(thresh, f"Action: {action}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))  # Draw the text
            # Draw the text
            cv2.putText(thresh, f"Prediction: {prediction} ({score}%)", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255))
            cv2.putText(thresh, f"Action: {action}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255))  # Draw the text
            cv2.imshow('ori', thresh)

            # get the contours
            thresh1 = copy.deepcopy(thresh)
            # _, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            length = len(contours)
            maxArea = -1
            if length > 0:
                for i in range(length):  # find the biggest contour (according to area)
                    temp = contours[i]
                    area = cv2.contourArea(temp)
                    if area > maxArea:
                        maxArea = area
                        ci = i

                res = contours[ci]
                hull = cv2.convexHull(res)
                drawing = np.zeros(img.shape, np.uint8)
                cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
                cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

            cv2.imshow('output', drawing)

        # Keyboard OP
        k = cv2.waitKey(10)
        if k == 27:  # press ESC to exit all windows at any time
            break;
        elif k == ord('b'):  # press 'b' to capture the background
            print('press b')
            global bgModel
            bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
            b.set_light(1, on_command)
            time.sleep(2)
            isBgCaptured = 1
            print('Background captured')
            pygame.init()
            pygame.mixer.init()


            # pygame.mixer.music.set_pos(50)
            # pygame.mixer.music.pause()

        elif k == ord('r'):  # press 'r' to reset the background
            time.sleep(1)
            bgModel = None
            triggerSwitch = False
            isBgCaptured = 0
            print('Reset background')
        elif k == 32:
            # If space bar pressed
            cv2.imshow('original', frame)
            # copies 1 channel BW image to all 3 RGB channels
            target = np.stack((thresh,) * 3, axis=-1)
            target = cv2.resize(target, (224, 224))
            # defalut setting value: 224
            target = target.reshape(1, 224, 224, 3)
            prediction, score = predict_rgb_image_vgg(target)

            if smart_home:
                if prediction == 'Palm':
                    try:
                        isalreadyfist = False
                        action = "X"
                        # sonos.play()
                        # pygame.mixer.music.unpause()
                    # Turn off smart home actions if devices are not responding
                    except ConnectionError:
                        smart_home = False
                        pass

                elif prediction == 'Fist':
                    try:
                        if isalreadyfist:
                            action = 'Lights OFF'
                            b.set_light(1, off_command)
                            isalreadyfist = False
                        else:
                            isalreadyfist = True

                    except ConnectionError:
                        smart_home = False
                        pass

                elif prediction == 'L':
                    try:
                        isalreadyfist = False
                        action = 'sound1'
                        pygame.mixer.music.load('./audio/headache.MP3')
                        pygame.mixer.music.play()

                    except ConnectionError:
                        smart_home = False
                        pass

                elif prediction == 'Okay':
                    try:
                        if isalreadyfist:
                            action = 'Lights ON'
                            b.set_light(1, on_command)
                            isalreadyfist = False
                        else:
                            pygame.mixer.music.load('./audio/wku.MP3')
                            pygame.mixer.music.play()
                    except ConnectionError:
                        smart_home = False
                        pass

                elif prediction == 'Peace':
                    try:
                        isalreadyfist = False
                        action = 'X'
                    except ConnectionError:
                        smart_home = False
                        pass

                else:
                    pass

            if save_images:
                img_name = f"./frames/drawings/drawing_{selected_gesture}_{img_counter}.jpg".format(
                    img_counter)
                cv2.imwrite(img_name, drawing)
                print("{} written".format(img_name))

                img_name2 = f"./frames/silhouettes/{selected_gesture}_{img_counter}.jpg".format(
                    img_counter)
                cv2.imwrite(img_name2, thresh)
                print("{} written".format(img_name2))

                img_name3 = f"./frames/masks/mask_{selected_gesture}_{img_counter}.jpg".format(
                    img_counter)
                cv2.imwrite(img_name3, img)
                print("{} written".format(img_name3))

                img_counter += 1


running = False


def run():
    global running
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    label.resize(int(width), int(height))
    while running:
        ret1, img1 = cap.read()
        if ret1:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            h, w, c = img1.shape
            qImg = QtGui.QImage(img1.data, w, h, w * c, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qImg)
            label.setPixmap(pixmap)
        else:
            QtWidgets.QMessageBox.about(win, "Error", "Cannot read frame.")
            print("cannot read frame.")
            break
    cap.release()
    print("Thread end.")


def stop():
    exit(0)


def start():
    global running
    running = True
    th = threading.Thread(target=run)
    th.start()
    print("started..")


def onExit():
    print("exit")
    stop()



app = QtWidgets.QApplication([])
win = QtWidgets.QWidget()
win.setStyleSheet("background-color: white;"
                  )
win.setFixedSize(640,960)
vbox = QtWidgets.QVBoxLayout()
label = QtWidgets.QLabel()
label.setFixedSize(620,540)
label.setStyleSheet("margin:50px;"
                    "border-style:solid; "
                    "border-color:black; border-width:5px; border-radius:10px;"
                    )
#shadow1=QGraphicsDropShadowEffect()
#shadow1.setBlurRadius(50)
#label.setGraphicsEffect(shadow1)
#label.setAlignment(Qt.AlignCenter)
# label.setStyleSheelt(QString("border-width: 2px, border-style: solid;"));
btn_start = QtWidgets.QPushButton("  CAMERA ON  ")
btn_start.setStyleSheet("color: white;"
                        "background-color: #0371f4;"  
                        "border-radius:28px;"
                        "font:20px;"
                        "border-width: 3px;"
                        "max-width:8em;"
                        #"margin:50px;"
                        "margin-left:180px;"
                        "padding: 15px;"
                        "font-family: Arial Black")
#shadow2=QGraphicsDropShadowEffect()
#shadow2.setBlurRadius(1)
#btn_start.setGraphicsEffect(shadow2)
btn_stop = QtWidgets.QPushButton("  CAMERA OFF  ")
btn_stop.setStyleSheet("color: white;"
                        "background-color: #0371f4;"  
                        "border-radius:28px;"
                        "font:20px;"
                        "border-width: 3px;"
                        "max-width:8em;"
                        #"margin:10px;"
                        "margin-left:180px;"
                        "padding: 15px;"
                        "font-family: Arial Black"
                        )
#shadow3=QGraphicsDropShadowEffect()
#shadow3.setBlurRadius(1)
#btn_stop.setGraphicsEffect(shadow3)
vbox.addWidget(label)
vbox.addWidget(btn_start)
vbox.addWidget(btn_stop)
win.setLayout(vbox)
win.show()



btn_start.clicked.connect(start)
btn_stop.clicked.connect(stop)
# app.aboutToQuit.connect(onExit)
run1()
sys.exit(app.exec_())

