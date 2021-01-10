#!/Users/kmihara/.pyenv/versions/3.7.4/envs/slamenv/bin/python
import cv2
import sdl2.ext
import numpy as np
np.set_printoptions(suppress=True)


from display import Display
from extractor import Extractor

W = 940
H = 580
F = 282

orb = cv2.ORB_create()
disp = Display(W,H)
K = np.array(([F,0,W//2], [0, F, H//2], [0, 0, 1]))
fe = Extractor(K)



def process_frame(img):
  img = cv2.resize(img, (W, H))
  #kp, des = fe.extract(img)
  matches = fe.extract(img)
  if matches is None:
      return
  #kp, des = orb.detectAndCompute(img, None)

  #for p in kp:
  #  #u, v = map(lambda x: int(round(x)), p.pt)
  #  u, v = map(lambda x: int(round(x)), p)
  #  cv2.circle(img, (u,v), color=(0,255,0), radius=3)
  #disp.paint(img)

  for pt1, pt2 in matches:
      u1, v1 = fe.denormalize(pt1)
      u2, v2 = fe.denormalize(pt2)
      #u1, v1 = map(lambda x: int(round(x)), pt1)
      #u2, v2 = map(lambda x: int(round(x)), pt2)


      cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
      cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0), thickness=2)
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

