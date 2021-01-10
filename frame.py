import cv2
import numpy as np

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform

#IRt = np.zeros((3,4))
#IRt[:, :3] = np.eye(3)
IRt = np.eye(4)


# turn [[x,y]] -> [[x, y, 1]]
def add_ones(x):
  return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def extractRt(E):
  W = np.mat([[0,-1,0],[1,0,0],[0,0,1]], dtype=float)
  U,d,Vt = np.linalg.svd(E)
  assert np.linalg.det(U) > 0
  if np.linalg.det(Vt) < 0:
    Vt *= -1.0
  R = np.dot(np.dot(U, W), Vt)
  if np.sum(R.diagonal()) < 0:
    R = np.dot(np.dot(U, W.T), Vt)
  t = U[:, 2]

  ret = np.eye(4)
  ret[:3, :3] = R
  ret[:3, 3] = t


  return ret


def extract(img):
  orb = cv2.ORB_create(1000)

  # detection
  pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 1000, qualityLevel=0.01, minDistance=3)

  # extraction
  kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
  kps, des = orb.compute(img, kps)

  #return pts, des
  #print("here")
  #print(np.array(pts).shape)
  return np.array([[kp.pt[0], kp.pt[1]] for kp in kps]), des
  #return np.array(pts).reshape(-1,2), des

def match_frames(f1, f2):
  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  matches = bf.knnMatch(f1.des, f2.des, k=2)

  # Lowe's ration test
  ret = []
  idx1, idx2 = [], []

  for m,n in matches:
    if m.distance < 0.75*n.distance:
      # keep around indices
      idx1.append(m.queryIdx)
      idx2.append(m.trainIdx)

      p1 = f1.pts[m.queryIdx]
      p2 = f2.pts[m.trainIdx]
      ret.append((p1, p2))

  assert len(ret) >= 8
  pose = None
  ret = np.array(ret)
  idx1 = np.array(idx1)
  idx2 = np.array(idx2)

  ## normalize coords: subtract to move to 0
  #ret[:,0,:] = normalize(f1.Kinv, ret[:,0,:])
  #ret[:,1,:] = normalize(f2.Kinv, ret[:,1,:])

  # filter and fit matrix
  model, inliers = ransac((ret[:, 0], ret[:, 1]),
                           EssentialMatrixTransform,
                           #FundamentalMatrixTransform,
                            min_samples=8, residual_threshold=0.005, max_trials=100)

  # ignore outliers
  ret = ret[inliers]
  Rt = extractRt(model.params)

  return idx1[inliers], idx2[inliers], Rt


def normalize(Kinv, pts):
  return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

def denormalize(K, pt):
  ret = np.dot(K, np.array([pt[0], pt[1], 1.0]).T)
  ret /= ret[2]
  return int(round(ret[0])), int(round(ret[1]))

class Frame(object):
  def __init__(self, img, K):
    self.K = K
    self.Kinv = np.linalg.inv(self.K)

    pts, self.des = extract(img)
    self.pts = normalize(self.Kinv, pts)
    self.pose = IRt


