#!/Users/kmihara/.pyenv/versions/3.7.4/envs/slamenv/bin/python

import cv2
import sdl2.ext
import numpy as np

from display import Display

W = 940
H = 580

disp = Display(W,H)


def process_frame(img):
    img = cv2.resize(img, (W, H))
    disp.paint(img)


if __name__ == "__main__":

    print("hello")
    cap = cv2.VideoCapture('drive.mp4')

    while(cap.isOpened()):
        ret, frame = cap.read()
        #window.refresh()
        if ret:
            process_frame(frame)
        else:
            break
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    #while(cap.isOpened()):
    #    ret, frame = cap.read()

    #    cv2.imshow('frame', frame)
    #    if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break

    #cap.release()
    #cv2.destroyAllWindows()
    #sdl2.ext.init()
    #window = sdl2.ext.Window("hello world", size = (640, 480))
    #window.show()
