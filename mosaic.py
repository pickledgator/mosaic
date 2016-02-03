#!/usr/bin/env python
#=========================================
# Project: Planar Mosaic Reconstruction
# Author:  N.Fischer
# Date:    2/3/16
#=========================================

import argparse
import os
import sys
import logging
import imutils
import cv2
import csv
import numpy as np
from numpy.linalg import inv

logging.basicConfig(format='%(asctime)s %(levelname)s [%(module)s] %(message)s', level=logging.INFO)

# images class preps data structure for use by Stitch class
# facilitates being able to handle multiple types of image gathering methods (load from S3, etc.)
class Images:
	def __init__(self):
		self.logger = logging.getLogger()
		self.imageList = []
		self.poseList = None
		self.imageWidth = 100
		self.imageHeight = 100
		self.filenames = []

	def loadFromDirectory(self, dirPath=None):
		self.logger.info("Searching for images and pose.csv in: {}".format(dirPath))

		if dirPath == None:
			raise Exception("You must specify a directory path to the source images")
		if not os.path.isdir(dirPath):
			raise Exception("Directory does not exist!")

		# grab pose data from csv
		self.poseList = self.getPoseData(dirPath)
		if len(self.poseList) == 0:
			self.logger.error("Error reading pose.csv")
			return False

		# grab filenames from directory
		self.filenames = self.getFilenames(dirPath)
		if self.filenames == None:
			self.logger.error("Error reading filenames, was directory empty?")
			return False

		# load the images
		for i,img in enumerate(self.filenames):
			self.logger.info("Opening file: {}".format(img))
			self.imageList.append(cv2.imread(img))
		
		# set attributes for images (based on image 1), assumes all images are the same size		
		(self.imageWidth, self.imageHeight) = self.getImageAttributes(self.imageList[0])

		self.logger.info("Data loaded successfully.")

	def getImageAttributes(self, img):
		return (img.shape[1], img.shape[0])

	def getPoseData(self, dirPath):
		# load pose data
		self.logger.info("Loading pose.csv...")
		reader = csv.DictReader(open(dirPath+'/pose.csv'))
		data = []
		for row in reader:
			for key,val in row.iteritems():
				val = val.replace('\xc2\xad', '') # some weird unicode characters in the list from pdf
				try:
					row[key] = float(val)
				except ValueError:
					row[key] = val
			data.append(row)
		self.logger.info("Read {} rows from pose.csv".format(len(data)))
		# helper dict for quickly finding pose data in O(1)
		poseByFilenameList = dict((d["filename"], dict(d, index=i)) for (i, d) in enumerate(data))
		return poseByFilenameList


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
			self.logger.info("Found {} files in directory: {}".format(len(filenames), sPath))
			return filenames

# stitch class executes keypoint extraction, matching and homography reconstruction to align images
# abstracted out data loading to facilitate easier unit testing, if desired
class Stitch:
	def __init__(self, imagesObj):
		self.logger = logging.getLogger()
		self.images = imagesObj.imageList
		self.poses = imagesObj.poseList
		self.imageWidth = imagesObj.imageWidth
		self.imageHeight = imagesObj.imageHeight
		self.filenames = imagesObj.filenames

	def rotateImageAndCenter(self, img, degreesCCW=0):
		scaleFactor = 1.0
		(oldY,oldX,oldC) = img.shape #note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
		M = cv2.getRotationMatrix2D(center=(oldX/2,oldY/2), angle=degreesCCW, scale=scaleFactor) #rotate about center of image.
		# choose a new image size.
		newX,newY = oldX*scaleFactor,oldY*scaleFactor
		# include this if you want to prevent corners being cut off
		r = np.deg2rad(degreesCCW)
		newX,newY = (abs(np.sin(r)*newY) + abs(np.cos(r)*newX),abs(np.sin(r)*newX) + abs(np.cos(r)*newY))
		# find the translation that moves the result to the center of that region.
		(tx,ty) = ((newX-oldX)/2,(newY-oldY)/2)
		M[0,2] += tx # third column of matrix holds translation, which takes effect after rotation.
		M[1,2] += ty 
		rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX),int(newY)))
		return rotatedImg

	def scaleAndCrop(self, img, outWidth):
		resized = imutils.resize(img, width=outWidth)
		grey = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
		ret,thresh = cv2.threshold(grey,10,255,cv2.THRESH_BINARY)
		out = cv2.findContours(thresh, 1, 2)
		cnt = out[0]
		x,y,w,h = cv2.boundingRect(cnt)
		crop = resized[y:y+h,x:x+w]
		return crop

	def initScaling(self, imageWidth, inScale, outScale):
		# compute scaling values for input and output images
		inWidth = int(imageWidth*inScale)
		windowSize = (inWidth*3,inWidth*3) # this should be a large canvas, used to create container size
		outWidth = int(windowSize[0]*outScale)
		windowShift = [inWidth/2, inWidth/2]
		self.logger.info("Scaling input image widths from {} to {}".format(imageWidth,inWidth))
		self.logger.info("Using canvas container width (input x2): {}".format(windowSize[0]))
		self.logger.info("Scaling output image width from {} to {}".format(windowSize[0],outWidth))
		return (inWidth, outWidth, windowSize, windowShift)

	def preprocessImages(self, poses, inWidth):
		# pre-process the images: resize and align by vehicle yaw (helps the matcher)
		for i,img in enumerate(self.images):
			self.images[i] = imutils.resize(self.images[i], width=inWidth) # reduce computation time
			poseI = poses[os.path.basename(self.filenames[i])] # get dict elements from filename
			yawI = poseI['yaw'] # already in degrees as needed by opencv rotation function
			self.logger.info("Rotating image {} by {} degrees".format(i, yawI))
			self.images[i] = self.rotateImageAndCenter(self.images[i],yawI)

	def process(self, ratio=0.75, reprojThresh=4.0, showMatches=False, outScale=1.0, inScale=1.0):
		# scale and rotate the input images accordingly
		(inWidth, outWidth, windowSize, windowShift) = self.initScaling(self.imageWidth, inScale, outScale)		
		self.preprocessImages(self.poses, inWidth)

		# extract the keypoints for each image frame
		kps = []
		features = []
		for i,img in enumerate(self.images):
			self.logger.info("Extracting SIFT features for input image {} of {}...".format(i+1,len(self.images)))
			(keypts, feats) = self.extractFeatures(img)
			kps.append(keypts)
			features.append(feats)

		# create some empty images for use in combining results
		base = np.zeros((windowSize[1],windowSize[0],3), np.uint8)
		container = np.array(base)
		# add base image to the new container
		base[windowShift[1]:self.images[0].shape[0]+windowShift[1], windowShift[0]:self.images[0].shape[1]+windowShift[0]] = self.images[0] # todo, handle arbitrary base image
		container = self.addImage(base, container, transparent=False)

		# find keypoints of newest container, run matching, apply transformation and stitch into container
		containerKpts = []
		containerFeats = []
		M = []
		H = []
		for i,img in enumerate(self.images[:-1]):
			# find keypoints of new container
			self.logger.info("Extracting SIFT features for container, iteration {} of {}...".format(i+1,len(self.images)-1))
			(containerKpts, containerFeats) = self.extractFeatures(container)

			# compute matches between container points and next image
			self.logger.info("Computing matches for image {} of {}...".format(i+1,len(self.images)-1))
			kpsMatches = self.matchKeypoints(kps[i+1], containerKpts, features[i+1], containerFeats, ratio, reprojThresh)
			if kpsMatches == None:
				self.logger.warning("kpsMatches == None!")
				continue
			M.append(kpsMatches)

			(matches, homography, status) = M[-1]
			H.append(homography)

			# apply transformation
			res = cv2.warpPerspective(self.images[i+1], H[-1], windowSize)	

			# add image to container
			container = self.addImage(res, container, transparent=False)
			# todo: better edge blending, pyramids, etc.
			
			# visualize intermediate steps, if desired
			if showMatches:
				vis = self.drawMatches(self.images[i+1], container, kps[i+1], containerKpts, matches, status)
				cv2.imshow("Matches", vis)
				cv2.waitKey(1)

		# scale the final output, and crop the container to remove excess blank space 
		# (container needs to be big during processing since the transformations may deviate from base image location in any direction)
		scaledContainer = self.scaleAndCrop(container, outWidth)
		
		# draw final scaled output
		cv2.imshow("Scaled Output",scaledContainer)
		self.logger.info("Hit space bar to close viewer...")
		cv2.waitKey(0)

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
		intersect = cv2.bitwise_and(threshImage, threshContainer) # find intersection between container and new image
		mask = cv2.subtract(threshImage,intersect) # subtract the intersection, leaving just the new part to union
		kernel = np.ones((3,3),'uint8') # for dilation below
		mask = cv2.dilate(mask,kernel,iterations=1) # make the mask slightly larger so we don't get blank lines on the edges
		maskedImage = cv2.bitwise_and(image, image, mask=mask) # apply mask
		con = cv2.add(container, maskedImage) # add the new pixels
		return con

	def extractFeatures(self, image):
		# convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# detect and extract features from the image
		descriptor = cv2.xfeatures2d.SIFT_create()
		(kps, features) = descriptor.detectAndCompute(image, None)
		self.logger.info("Found {} keypoints in frame".format(len(kps)))

		# convert the keypoints from KeyPoint objects to np
		kps = np.float32([kp.pt for kp in kps])

		# return a tuple of keypoints and features
		return (kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
		# compute the raw matches and build list of actual matches that pass check

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
	ap.add_argument("-os", "--outscale", default=1.0, type=float, help="ratio by which to scale the output image")
	ap.add_argument("-is", "--inscale", default=1.0, type=float, help="ratio by which to scale the input images (faster processing)")
	ap.add_argument("-m", "--showmatches", action='store_true', help="show intermediate matches and container")
	args = vars(ap.parse_args())

	imgs = Images()
	imgs.loadFromDirectory(args['dir'])

	mosaic = Stitch(imgs)
	mosaic.process(ratio=0.75, reprojThresh=4.0, showMatches=args['showmatches'], outScale=args['outscale'], inScale=args['inscale'])

