#!/Users/kmihara/.pyenv/versions/3.7.4/envs/slamenv/bin/python
import cv2
import sdl2.ext
import numpy as np
np.set_printoptions(suppress=True)
import g2o

from display import Display
from frame import Frame, denormalize, match_frames
# camera intrinsic
W = 940
H = 580
F = 300

# main classes
orb = cv2.ORB_create()
disp = Display(W,H)
K = np.array(([F,0,W//2], [0, F, H//2], [0, 0, 1]))
IRt = np.zeros((3,4))
IRt[:, :3] = np.eye(3)
#fe = Extractor(K)

class Point(object):
  # A Poing is a 3-D point in the world
  # Each point is observed in multiple frames
  def __init__(self, loc):
    self.frames = []
    self.location = loc
    self.idxes = []

  def add_observation(self, frame, idx):
    self.frames.append(frame)
    self.idxes.append(idx)

def triangulate(pose1, pose2, pts1, pts2):
    return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T


frames = []

def process_frame(img):
  img = cv2.resize(img, (W, H))
  frame = Frame(img, K)
  frames.append(frame)

  if len(frames) <= 1:
    return

  idx1, idx2, Rt = match_frames(frames[-1], frames[-2])
  frames[-1].pose = np.dot(Rt, frames[-2].pose)

  # homogeneous 3-D coords
  pts4d = triangulate(frames[-1].pose, frames[-2].pose, frames[-1].pts[idx1], frames[-2].pts[idx2])
  pts4d /= pts4d[:, 3:]


  # reject pts without enough "parallax"
  # reject points behind the camera
  good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:,2] > 0)
  for i, p in enumerate(pts4d):
    if not good_pts4d[i]:
      continue
    pt = Point(p)
    pt.add_observation(frames[-1], idx1[i])
    pt.add_observation(frames[-2], idx2[i])


  for pt1, pt2 in zip(frames[-1].pts[idx1], frames[-2].pts[idx2]):
      u1, v1 = denormalize(frame.K, pt1)
      u2, v2 = denormalize(frame.K, pt2)
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

