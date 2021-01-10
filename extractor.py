import cv2
import numpy as np

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform

# turn [[x,y]] -> [[x, y, 1]]
def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

class Extractor(object):
  GX = 16//2
  GY = 12//2
  def __init__(self, K):
    self.orb = cv2.ORB_create(1000)
    self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    self.last = None
    self.K = K 
    self.Kinv = np.linalg.inv(self.K)

  def normalize(self, pts):
      return np.dot(self.Kinv, add_ones(pts).T).T[:, 0:2]

  def denormalize(self, pt):
      ret = np.dot(self.K, np.array([pt[0], pt[1], 1.0]))
      #ret /= ret[2]
      return int(round(ret[0])), int(round(ret[1]))
      #return int(round(pt[0] + self.w)), int(rount(pt[1] + self.h))


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

      # normalize coords: subtract to move to 0
      ret[:,0,:] = self.normalize(ret[:,0,:])
      ret[:,1,:] = self.normalize(ret[:,1,:])
      #ret[:,1,:] = np.dot(self.Kinv, add_ones(ret[:,1,:]).T).T[:,0:2]
      #ret[:, :, 0] -= img.shape[0]//2
      #ret[:, :, 1] -= img.shape[1]//2
  #print(np.shape(ret))

    # filter
      model, inliers = ransac((ret[:, 0], ret[:, 1]),
                        #EssentialMatrixTransform,
                            FundamentalMatrixTransform,
                             min_samples=8, residual_threshold=1, max_trials=100)
      ret = ret[inliers]

      s,v,d = np.linalg.svd(model.params)


    self.last = {'kps': kps, 'des': des}
      #print(matches)

    #return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des
    #return matches
    return ret
    #return kps, des

