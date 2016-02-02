#!/usr/bin/env python

import argparse
import os
import sys
import logging
import numpy as np
from numpy.linalg import inv
import imutils
import cv2
import csv

logging.basicConfig(format='%(asctime)s %(levelname)s [%(module)s] %(message)s', level=logging.INFO)

class Stitch:
	def __init__(self, dirPath=None):
		self.logger = logging.getLogger()
		if dirPath == None:
			self.logger.error("You must specify a directory path to the source images!")
			return
		if not os.path.isdir(dirPath):
			self.logger.error("Directory does not exist!")
			return
		self.dirPath = dirPath

		# grab filenames from directory
		self.filenames = self.getFilenames(self.dirPath)
		if self.filenames == None:
			self.logger.error("Error reading filenames, was directory empty?")
			return

		# create cvimage objects
		self.images = []
		for i in self.filenames:
			self.images.append(cv2.imread(i))
			self.images[-1] = imutils.resize(self.images[-1], width=400)
			self.images[-1] = cv2.copyMakeBorder(self.images[-1],20,20,20,20,cv2.BORDER_CONSTANT,0)
			rot = cv2.getRotationMatrix2D((self.images[-1].shape[1],self.images[-1].shape[0]),10,1)
			dst = cv2.warpAffine(self.images[-1],rot,(self.images[-1].shape[1],self.images[-1].shape[0]))
			cv2.imshow("Ori",self.images[-1])
			cv2.imshow("Rot",dst)
			cv2.waitKey(0)

		# load pose data
		self.logger.info("Loading pose.csv...")
		reader = csv.DictReader(open(self.dirPath+'/pose.csv'))
		poseData = []
		for row in reader:
			poseData.append(row)

		# init other variable lists
		self.kps = []
		self.features = []
		self.M = []
		self.H = []
		self.results = []
		self.shift = [200,200]
		self.windowSize = (1000,1000)

	def getFilenames(self, sPath):
		filenames = []
		for sChild in os.listdir(sPath):     
			# check for valid file types here    
			if os.path.splitext(sChild)[1][1:] in ['jpg', 'png']:       
				sChildPath = os.path.join(sPath,sChild)
				filenames.append(sChildPath)
		if len(filenames) == 0:
			return None
		else:
			self.logger.info("Read {} files from directory: {}".format(len(filenames), sPath))
	    	return filenames

	def process(self, ratio=0.75, reprojThresh=4.0, showMatches=False):

		for img in self.images:
			(keypts, feats) = self.extractFeatures(img)
			self.kps.append(keypts)
			self.features.append(feats)

		for i,img in enumerate(self.images[:-1]):
			kpsMatches = self.matchKeypoints(self.kps[i+1], self.kps[i], self.features[i+1], self.features[i], ratio, reprojThresh)
			if kpsMatches == None:
				continue
			self.M.append(kpsMatches)

			(matches, homography, status) = self.M[-1]
			self.H.append(homography)
			shiftH = np.array([[1,0,self.shift[0]],[0,1,self.shift[1]],[0,0,1]], np.float64)
			chainedH = np.array([[1,0,0],[0,1,0],[0,0,1]], np.float64)
			chainedH = shiftH.dot(chainedH)
			for h in reversed(self.H):
				chainedH = chainedH.dot(h)
			res = cv2.warpPerspective(self.images[i+1], chainedH, self.windowSize)
			self.results.append(res)
			vis = self.drawMatches(self.images[i+1], self.images[i], self.kps[i+1], self.kps[i], matches, status)
			# cv2.imshow("Matches", vis)
			# cv2.waitKey(0)

		# create some empty images for use in combining results
		base = np.zeros((self.windowSize[1],self.windowSize[0],3), np.uint8)
		container = np.array(base)

		# add base image
		base[self.shift[1]:self.images[0].shape[0]+self.shift[1], self.shift[0]:self.images[0].shape[1]+self.shift[0]] = self.images[0]
		container = self.addImage(base,container, transparent=False)

		# add other images that have been warped
		for res in self.results:
			container = self.addImage(res, container, transparent=False)

		# check to see if the keypoint matches should be visualized
		#if showMatches:
			#for i,image in enumerate(self.images):
			#	cv2.imshow("Image {}".format(i), image)
			#cv2.imshow("Result", container)
			#cv2.waitKey(0)

		# return the stitched image
		return True

	def addImageSlow(self, image, container):
		for (x,y,c), value in np.ndenumerate(image):
			print("{} {} {} {}".format(x,y,c,value))
			if container[x,y,c] > 0 and value > 0:
				self.logger.info("Setting {} at {},{},{}".format(value,x,y,c))
				container.itemset((x,y,c),value)
		return container

	def addImage(self, image, container, first=False, transparent=False):
		if transparent:
			con = cv2.addWeighted(container, 0.5, image, 0.5, 0.0)
			cv2.imshow("Container", con)
			cv2.waitKey(0)
			return con

		# if the container is empty, just return the full image
		if first:
			return image
		# else threshold both images, find non-overlapping sections, add to container
		greyImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		greyContainer = cv2.cvtColor(container,cv2.COLOR_BGR2GRAY)
		ret,threshImage = cv2.threshold(greyImage,10,255,cv2.THRESH_BINARY)
		ret,threshContainer = cv2.threshold(greyContainer,10,255,cv2.THRESH_BINARY)
		intersect = cv2.bitwise_and(threshImage, threshContainer)
		mask = cv2.subtract(threshImage,intersect)
		maskedImage = cv2.bitwise_and(image, image, mask=mask)
		con = cv2.add(container, maskedImage)
		cv2.imshow("Container", con)
		cv2.waitKey(0)
		#res = cv2.add(image, container)
		#res = res / 2.0
		return con


	def extractFeatures(self, image):
		# convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# detect and extract features from the image
		self.logger.info("Running SIFT, extracting features...")
		descriptor = cv2.xfeatures2d.SIFT_create()
		(kps, features) = descriptor.detectAndCompute(image, None)
		self.logger.info("Found {} keypoints in frame".format(len(kps)))

		# convert the keypoints from KeyPoint objects to np
		kps = np.float32([kp.pt for kp in kps])

		# return a tuple of keypoints and features
		return (kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
		# compute the raw matches and initialize the list of actual
		# matches
		self.logger.info("Computing matches...")

		# FLANN_INDEX_KDTREE = 0
		# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		# search_params = dict(checks = 50)   # or pass empty dictionary
		# flann = cv2.FlannBasedMatcher(index_params,search_params)
		# rawMatches = flann.knnMatch(featuresA,featuresB,k=2)

		matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

		self.logger.info("Found {} raw matches".format(len(rawMatches)))

		matches = []
		# loop over the raw matches and remove outliers
		for m in rawMatches:
			# ensure the distance is within a certain ratio of each
			# other (i.e. Lowe's ratio test)
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))

		self.logger.info("Found {} matches after Lowe's test".format(len(matches)))

		# computing a homography requires at least 4 matches
		if len(matches) > 4:
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])

			# compute the homography between the two sets of points
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

			# return the matches along with the homograpy matrix
			# and status of each matched point
			return (matches, H, status)

		# otherwise, no homograpy could be computed
		self.logger.warning("Homography could not be computed!")
		return None

	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# initialize the output visualization image
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB

		# loop over the matches
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully
			# matched
			if s == 1:
				# draw the match
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

		# return the visualization
		return vis

if __name__=="__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dir", required=True, help="directory of images (jpg, png)")
	args = vars(ap.parse_args())

	mosaic = Stitch(args['dir'])
	#res = mosaic.process(showMatches=True)
	#if not res:
	#	mosaic.logger.error('Mosaic failed.')




