import scipy.io
import cv2
import numpy as np

import sys
sys.path.append('./src')
import RANSAC
import BASE_utils


def draw_match(img1, img2, corr1, corr2):

    corr1 = [cv2.KeyPoint(corr1[i, 0], corr1[i, 1], 1) for i in range(corr1.shape[0])]
    corr2 = [cv2.KeyPoint(corr2[i, 0], corr2[i, 1], 1) for i in range(corr2.shape[0])]

    assert len(corr1) == len(corr2)

    draw_matches = [cv2.DMatch(i, i, 0) for i in range(len(corr1))]

    display = cv2.drawMatches(img1, corr1, img2, corr2, draw_matches, None,
                              matchColor=(0, 255, 0),
                              singlePointColor=(0, 0, 255),
                              flags=0
                              )
    return display

if __name__ == "__main__":

    data = scipy.io.loadmat('data/box.mat')
    X = data['X0']
    Y = data['Y0']
    I1 = data['I1']
    I2 = data['I2']
    vpts = data['vpts']

    # This is to avoid some annoying warnings that don't bother the results
    np.seterr(divide='ignore', invalid='ignore')

    maxTrials = 5000            # maximum number of trials
    th = 2.0                    # inlier-outlier threshold in pixels
    confidence = 0.999999       # confidence value
    DEGEN = True                # degeneracy updating
    LO = True                   # local optimization
    F, mask = RANSAC.findFundamentalMatrix(X, Y, maxTrials, th, confidence, DEGEN, LO)
    print("Mean Geometric Error: {} pixels".format(np.mean(BASE_utils.sampsonDistanceF(vpts[0:3,:],vpts[3:6,:],F))))

    display = draw_match(I1, I2, X[mask,:], Y[mask,:])
    cv2.imshow("fundamental matrix estimation visualization", display)
    print('please press any key to terminate window')
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

    data = scipy.io.loadmat('data/graf.mat')
    X = data['X0']
    Y = data['Y0']
    I1 = data['I1']
    I2 = data['I2']
    vpts = data['vpts']

    maxTrials = 5000
    th = 1.0
    confidence = 0.999999
    LO = True
    H, mask = RANSAC.findHomography(X, Y, maxTrials, th, confidence, LO)
    print("Mean Geometric Error: {} pixels".format(np.mean(BASE_utils.sampsonDistanceH(vpts[0:3,:],vpts[3:6,:],H))))

    display = draw_match(I1, I2, X[mask,:], Y[mask,:])
    cv2.imshow("homography estimation visualization", display)
    print('please press any key to terminate window')
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

