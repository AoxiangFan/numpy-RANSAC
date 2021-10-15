import numpy as np
import BASE_utils
import LO_utils
import DEGEN_utils

def findFundamentalMatrix(X, Y, maxTrials, threshold, confidence, DEGEN, LO):
	N = X.shape[0]
	X = np.hstack((X,np.ones((N,1)))).T
	Y = np.hstack((Y,np.ones((N,1)))).T
	curTrials = 0
	bestInliers = []
	numBestInliers = 0

	numGoodInliers = 0

	BASE_utils.preSampsonDistanceH_all(X, Y)

	while curTrials <= maxTrials:
		F, indices, curInliers = BASE_utils.minimalSampleF(X, Y, threshold)
		numCurInliers = len(curInliers)
		if numGoodInliers < numCurInliers:
			numGoodInliers = numCurInliers
			maxTrials = BASE_utils.updateMaxTrials(numGoodInliers, maxTrials, N, confidence, 7)
			if numBestInliers < numCurInliers:
				bestInliers = curInliers
				numBestInliers = numCurInliers
			if LO:
				if numCurInliers >= 16:
					F, curInliers = LO_utils.LO_F(X, Y, F, threshold, 3, 50)
					numCurInliers = len(curInliers)
					if numBestInliers < numCurInliers:
						bestInliers = curInliers
						numBestInliers = numCurInliers
			if DEGEN:
				flag, H = DEGEN_utils.checkSample(X[:,indices], Y[:,indices], F, 2*threshold)
				if flag:
					dH = BASE_utils.sampsonDistanceH_all(X, Y, H)
					inliersH = np.where(dH<=3*threshold)[0]
					if len(inliersH) >= 8:
						H, inliersH = LO_utils.LO_H(X, Y, H, threshold, 3, 50)
						if len(inliersH) >= 6:
							F, inliersF = DEGEN_utils.H2F(X, Y, H, threshold)
							if numBestInliers < len(inliersF):
								bestInliers = inliersF
								numBestInliers = len(inliersF)
		curTrials = curTrials + 1
	if len(bestInliers) >= 8:
		F = BASE_utils.norm8Point(X[:, bestInliers], Y[:, bestInliers])
		d = BASE_utils.sampsonDistanceF(X, Y, F)
		bestInliers = np.where(d<=threshold)[0]
	else:
		F = np.ones((3,3))

	return F, bestInliers



def findHomography(X, Y, maxTrials, threshold, confidence, LO):
	N = X.shape[0]
	X = np.hstack((X,np.ones((N,1)))).T
	Y = np.hstack((Y,np.ones((N,1)))).T
	curTrials = 0
	bestInliers = []
	numBestInliers = 0

	numGoodInliers = 0

	BASE_utils.preSampsonDistanceH_all(X, Y)
	
	while curTrials <= maxTrials:
		H, _ = BASE_utils.minimalSampleH(X, Y)
		d = BASE_utils.sampsonDistanceH_all(X, Y, H)
		curInliers = np.where(d<=threshold)[0]
		numCurInliers = len(curInliers)
		if numGoodInliers < numCurInliers:
			numGoodInliers = numCurInliers
			maxTrials = BASE_utils.updateMaxTrials(numGoodInliers, maxTrials, N, confidence, 4)
			if numBestInliers < numCurInliers:
				bestInliers = curInliers
				numBestInliers = numCurInliers
			if LO:
				if numCurInliers >= 8:
					H, curInliers = LO_utils.LO_H(X, Y, H, threshold, 3, 50)
					numCurInliers = len(curInliers)
					if numBestInliers < numCurInliers:
						bestInliers = curInliers
						numBestInliers = numCurInliers
		curTrials = curTrials + 1
		
	if len(bestInliers) >= 4:
		H = BASE_utils.norm4Point(X[:, bestInliers], Y[:, bestInliers])
		d = BASE_utils.sampsonDistanceH_all(X, Y, H)
		bestInliers = np.where(d<=threshold)[0]
	else:
		H = np.ones((3,3))

	return H, bestInliers