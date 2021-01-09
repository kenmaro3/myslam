#!/Users/kmihara/.pyenv/versions/3.7.4/envs/slamenv/bin/python

import cv2
import sdl2.ext
import numpy as np

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

from display import Display

W = 940
H = 580

orb = cv2.ORB_create()
disp = Display(W,H)

class FeatureExtractor(object):
  GX = 16//2
  GY = 12//2
  def __init__(self):
    self.orb = cv2.ORB_create(1000)
    self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    self.last = None


  def extract(self, img):

    # detection
    pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 1000, qualityLevel=0.01, minDistance=3)

    # extraction 
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
    kps, des = self.orb.compute(img, kps)

    # maching
    ret = []
    matches = None
    if self.last is not None:
      #matches = self.bf.match(des, self.last['des'])
      matches = self.bf.knnMatch(des, self.last['des'], k=2)
      for m,n in matches:
        if m.distance < 0.75*n.distance:
          kp1 = kps[m.queryIdx].pt
          kp2 = self.last['kps'][m.trainIdx].pt
          ret.append((kp1, kp2))


    if len(ret) > 0:
      ret = np.array(ret)
      #print(np.shape(ret))

    # filter
      model, inliers = ransac((ret[:, 0], ret[:, 1]),
                            FundamentalMatrixTransform,
                             min_samples=8, residual_threshold=1, max_trials=100)
      ret = ret[inliers]

      print(sum(inliers))

    self.last = {'kps': kps, 'des': des}
      #print(matches)

    #return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des
    #return matches
    return ret
    #return kps, des


fe = FeatureExtractor()



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
      u1, v1 = map(lambda x: int(round(x)), pt1)
      u2, v2 = map(lambda x: int(round(x)), pt2)
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

