#!/usr/bin/env python

import argparse
import os
import sys
import logging
import numpy as np
from numpy.linalg import inv
import imutils
import cv2

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


		# imageA = cv2.imread(self.images[0])
		# imageB = cv2.imread(self.images[1])
		# imageA = imutils.resize(imageA, width=400)
		# imageB = imutils.resize(imageB, width=400)
		# imageA = self.images[0]
		# imageB = self.images[1]
		# imageC = self.images[2]

		# unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
		#(imageB, imageA) = images
		# (kpsA, featuresA) = self.extractFeatures(imageA)
		# (kpsB, featuresB) = self.extractFeatures(imageB)
		# (kpsC, featuresC) = self.extractFeatures(imageC)
		for img in self.images:
			(keypts, feats) = self.extractFeatures(img)
			self.kps.append(keypts)
			self.features.append(feats)

		for i,img in enumerate(self.images[:-1]):
			kpsMatches = self.matchKeypoints(self.kps[i+1],self.kps[i], self.features[i+1], self.features[i], ratio, reprojThresh)
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

		# match features between the two images
		# M = self.matchKeypoints(kpsB, kpsA, featuresB, featuresA, ratio, reprojThresh)
		# M2 = self.matchKeypoints(kpsC, kpsB, featuresC, featuresB, ratio, reprojThresh)

		# if the match is None, then there aren't enough matched
		# keypoints to create a panorama
		# if M is None:
			# return False

		# if M2 is None:
		# 	return False

		# otherwise, apply a perspective warp to stitch the images
		# together
		# (matches, H, status) = M
		# (matches2, H2, status2) = M2
		# shift = (0,0)
		# Ht = np.array([[1,0,shift[0]],[0,1,shift[1]],[0,0,1]], np.float64)
		# result = cv2.warpPerspective(imageB, Ht.dot(H), (1000,600))
		#result[shift[1]:imageA.shape[0]+shift[1], shift[0]:imageA.shape[1]+shift[0]] = imageA
		# result2 = cv2.warpPerspective(imageC, Ht.dot(H2.dot(H)), (1000,600))
		#result2[shift[1]:imageA.shape[0]+shift[1], shift[0]:imageA.shape[1]+shift[0]] = imageA

		# create some empty images for use in combining results
		base = np.zeros((self.windowSize[1],self.windowSize[0],3), np.uint8)
		container = np.array(base)

		# add base image
		base[self.shift[1]:self.images[0].shape[0]+self.shift[1], self.shift[0]:self.images[0].shape[1]+self.shift[0]] = self.images[0]
		container = self.addImage(base,container)
		#container = cv2.addWeighted(container, 0.5, base, 0.5, 0.0)
		# add other images that have been warped
		for res in self.results:
			container = self.addImage(res,container)
			# container = cv2.addWeighted(container, 0.5, res, 0.5, 0.0)

		# check to see if the keypoint matches should be visualized
		if showMatches:
			for i,image in enumerate(self.images):
				cv2.imshow("Image {}".format(i), image)
			# vis = self.drawMatches(imageB, imageA, kpsB, kpsA, matches,status)
			# cv2.imshow("Keypoint Matches", vis)
			cv2.imshow("Result", container)
			cv2.waitKey(0)

			# return a tuple of the stitched image and the
			# visualization
			# return (result, vis)

		# return the stitched image
		return True

	def addImageSlow(self, image, container):
		for (x,y,c), value in np.ndenumerate(image):
			print("{} {} {} {}".format(x,y,c,value))
			if container[x,y,c] > 0 and value > 0:
				self.logger.info("Setting {} at {},{},{}".format(value,x,y,c))
				container.itemset((x,y,c),value)
		return container

	def addImage(self, image, container):
		res = cv2.add(image, container)
		#res = res / 2.0
		return res


	def extractFeatures(self, image):
		# convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# check to see if we are using OpenCV 3.X
		# if self.isv3:
		# detect and extract features from the image
		self.logger.info("Running SIFT, extracting features...")
		descriptor = cv2.xfeatures2d.SIFT_create()
		(kps, features) = descriptor.detectAndCompute(image, None)
		self.logger.info("Found {} keypoints in frame".format(len(kps)))

		# # otherwise, we are using OpenCV 2.4.X
		# else:
		# 	# detect keypoints in the image
		# 	detector = cv2.FeatureDetector_create("SIFT")
		# 	kps = detector.detect(gray)

		# 	# extract features from the image
		# 	extractor = cv2.DescriptorExtractor_create("SIFT")
		# 	(kps, features) = extractor.compute(gray, kps)

		# convert the keypoints from KeyPoint objects to NumPy
		# arrays
		kps = np.float32([kp.pt for kp in kps])

		# return a tuple of keypoints and features
		return (kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
		# compute the raw matches and initialize the list of actual
		# matches
		self.logger.info("Computing matches...")
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
	res = mosaic.process(showMatches=True)
	if not res:
		mosaic.logger.error('Mosaic failed.')




