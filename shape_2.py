import cv2
import matplotlib.pyplot as plt
import imutils
import numpy as np

import sys # for command line arguments

#ok I don't even need this. Each coord could just be stored in a numpy array
class coord(object):
	"""docstring for coord"""
	def __init__(self, x, y):
		self.x = x
		self.y = y
	# def __init__(self, coords):
	# 	self.x = coords[0]
	# 	self.y = coords[1]

	#operator overloading:
	def __str__(self):
		return("(" + str(self.x) + ", " + str(self.y) + ")")
	def __repr__(self):
   		return str(self)
	def __eq__(self, other):
		return(self.x == other.x and self.y == other.y)


	# member functions
	def findSlope(self, other):
		dy = self.y - other.y
		dx = self.x - other.x

		if (dx == 0):
			slope  = 999999999999999999999
		else:
			slope = float(dy)/dx

		deriv = coord(slope, self.x)
		return deriv

	def asArray(self): #eugh...
		return [self.x, self.y]

# Returns an array of contours. Each contour corresponds to a found card.
# Also used to find the contours of the shapes within each card.
# May have to go in and change the threshold value. Seems to change with lighting.
def createContours(image):

	# preprocess the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	median_blurred = cv2.medianBlur(gray,21)

	thresh = cv2.threshold(median_blurred, 150, 255, cv2.THRESH_BINARY)[1] #may want to chose a different thresh
	# plt.subplot(1, 2, 1),plt.imshow(median_blurred),plt.axis('off'),plt.title('blurred')
	# plt.subplot(1, 2, 2),plt.imshow(thresh),plt.axis('off'),plt.title('thresh')
	# plt.show()




	# find contours in the thresholded image
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1] #version check?

	# loop over the contours
	for c in cnts:
		# compute the center of the contour
		# print "contour is", c
		M = cv2.moments(c)
		cX = 0
		cY = 0
		if (M["m00"] != 0):
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])

		# cv2.drawContours(image, c, -1, (0,255,0), 20)

	boundingRects = [cv2.boundingRect(c) for c in cnts] #returns x,y,width,height (x,y) is top left
	boundingRects_points = []
	masks = [image[b[1]:b[1]+b[3], b[0]:b[0]+b[2]] for b in boundingRects]
	
	return cnts

#contour is a numpy array
# only good where images are aligned with frame. Could fix by finding 4 max in second derivative of the contour
def findContourCorners(contour, image):
	topright = -1000000
	tr = coord(0,0)
	topleft = -1000000
	tl = coord(0,0)
	bottomleft = -1000000
	bl = coord(0,0)
	bottomright = -1000000
	br = coord(0,0)

	for point in contour:
		# print "point is", point
		x,y = point[0][0],point[0][1]
		# print "    ",x,y
		if (x + y >= topright):
			tr = coord(x,y)
			topright = x + y
		if (x - y >= bottomright):
			br = coord(x,y)
			bottomright = x - y
		if (-1*x + y >= topleft):
			tl = coord(x,y)
			topleft = y - x
		if (-1*x - y >= bottomleft):
			bl = coord(x,y)
			bottomleft = -1*x - y

	# cv2.drawContours(image, [contour], -1, (0,255,0), 2)

	# circleRadius = 5
	# cv2.circle(image, (tr.x, tr.y), circleRadius, (255, 0, 0), -1)
	# cv2.circle(image, (tl.x, tl.y), circleRadius, (255, 255, 0), -1)
	# cv2.circle(image, (br.x, br.y), circleRadius, (255, 0, 255), -1)
	# cv2.circle(image, (bl.x, bl.y), circleRadius, (0, 255, 255), -1)

	return [[tl.x, tl.y], [tr.x, tr.y], [bl.x, bl.y], [br.x, br.y]] #return the coordinates in [] form

def findContourCorners2(contour, image): #done with second derivative

	conts = []
	d1_conts = [] # y' , x ===> y'' = y'_1 - y'2 / x_1 - x_2
	d2_conts = [] # y'' , x
	for point in contour:
		x,y = point[0][0], point[0][1]
		c = coord(x,y)
		conts.append(c)


	pointShift = 2 # how many points to we want to skip before calculating the slope?

	for idx, point in enumerate(conts): #find slope of current point and the one after the next one.
		nextPoint = (idx+pointShift) % len(conts) # loop around if we go over
		slope = conts[idx].findSlope(conts[nextPoint])
		d1_conts.append(slope)

	# print d1_conts
	for idx, d1 in enumerate(d1_conts):
		nextPoint = (idx+pointShift) % len(d1_conts) # loop around if we go over
		slope = d1_conts[idx].findSlope(d1_conts[nextPoint])
		d2_conts.append(slope)

	for s in d2_conts:
		print "slope slope is", s

	#find the four highest magnitude values of d2

	highest_d2 = [0,0,0,0]

	highest_coords = [coord(0,0) for i in range(4)]

	d2_conts_y_val = [val.x for val in d2_conts]

	for idx, d2 in enumerate(d2_conts_y_val):
		t = 0
		biggerFound = False
		while (t < len(highest_d2) and abs(d2) > abs(highest_d2[t])):
			t += 1
			biggerFound = True
		if (biggerFound == True):
			highest_d2[t-1] = d2
			highest_coords[t-1] = conts[idx]

	print highest_d2
	print highest_coords

	circleRadius = 5
	cv2.circle(image, (highest_coords[0].x, highest_coords[0].y), circleRadius, (255, 0, 0), -1)
	cv2.circle(image, (highest_coords[1].x, highest_coords[1].y), circleRadius, (255, 255, 0), -1)
	cv2.circle(image, (highest_coords[2].x, highest_coords[2].y), circleRadius, (255, 0, 255), -1)
	cv2.circle(image, (highest_coords[3].x, highest_coords[3].y), circleRadius, (0, 255, 255), -1)

# Makes a card rectangular.
def fixPerspective(contour, image):
	# -----------------------------------------------
	# Fix the perspective.                      
	# -----------------------------------------------
	# get image dimensions
	height, width, channels = image.shape
	
	# set source and dest_rect
	source = findContourCorners(contour, image)

	# dest_rect = tl, tr, bl, br
	dest_rect = [[0,0], [width, 0], [0,height], [width, height]]

	pts1 = np.float32(source)
	pts2 = np.float32(dest_rect)

	M = cv2.getPerspectiveTransform(pts1,pts2)
	dst = cv2.warpPerspective(img,M,(width,height))

	# -----------------------------------------------
	# Correct White Balance                       
	# -----------------------------------------------
	# crop the image:
	height, width, channels = dst.shape
	shave_width = 20 # the number of pixels to shave off each edge
	dst = dst[shave_width:height - shave_width, shave_width:width-shave_width]
	# white = filter_increaseColor_hsv(dst)
	white = dst ##############################################################NEED TO FILTER MAYBE
	# -----------------------------------------------
	# Display the images.                       
	# -----------------------------------------------
	# plt.subplot(1, 2, 1),plt.imshow(dst),plt.axis('off'),plt.title('dst')
	# plt.subplot(1, 2, 2),plt.imshow(white_inv),plt.axis('off'),plt.title('white')
	# plt.show()

	return white

# -----------------------------------------------
# Filter functions
# Various Filters that should allow better processing of the image. HSV works best to remove gray.                   
# -----------------------------------------------
def filter_increaseColor_hsv(img):
	# Convert BGR to HSV
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# define range of color in HSV
	# we want to let through any hue, and any value. We want to remove the white with low saturation
	lower_color = np.array([0,15,0])
	upper_color = np.array([255,255,255])
	# Threshold the HSV image to get only blue colors
	mask = cv2.inRange(hsv, lower_color, upper_color)
	inv_mask = cv2.bitwise_not(mask)
	# Bitwise-AND mask and original image
	result = cv2.bitwise_and(img,img, mask= mask)
	print mask.shape
	black = cv2.bitwise_xor(img,img)
	white = cv2.bitwise_not(black)
	final = result + white

	#dont have to convert back to rgb?

	return final

def filter_increaseColor_bgr(img):

	copy = img.copy()

	# shape looks like(469, 760, 3) (number of y pixels, number of x pixels, number of channels)
	# for any pixel, if r = g = b, set it to 255, 255, 255. Hopefully this will make the gray go away.
	for  y_idx, y_val in enumerate(img):
		for x_idx, pixel in enumerate(y_val):
			if (roughlyEqual(pixel,10) == True):
				copy[y_idx][x_idx][:] = 255

	return copy
# returns true if vals are within some threshold of their average
def roughlyEqual(vals, thresh):
	avg = sum(vals)/len(vals)
	for val in vals:
		if (abs(val - avg) > thresh):
			return False
	return True

def identifyFeatures(img):
	cnts = createContours(img)

	height, width, channels = img.shape

	# img = img[50:height-50, 50:width-50]  # crop the image
	img_inv = cv2.bitwise_not(img)
	# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

	# preprocess the image
	gray = cv2.cvtColor(img_inv, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)[1]
	# find contours in the thresholded image
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1] #version check?

	#compare the area of the bounding rectangle with the area of the contour
	extents = []
	total_area = 0


	# clean up stuff outside of the contours we just detected.
	filled_contours = img.copy()
	for c in cnts:
		cv2.fillConvexPoly(filled_contours, c, (0,0,0)) # fill with black to use for mask later
	gray = cv2.cvtColor(filled_contours, cv2.COLOR_BGR2GRAY)
	mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1] #remove everything that isn't exactly black
	mask = cv2.bitwise_not(mask)
	img_clean = cv2.bitwise_and(img, img, mask= mask)

	# --- Detect Shape --- #
	for c in cnts:
		area = cv2.contourArea(c)
		total_area += area
		x,y,w,h = cv2.boundingRect(c)
		rect_area = w*h
		extents.append(float(area)/rect_area)

	avg_extents = sum(extents)/len(extents)

	shape = decide_shape_from_extent(avg_extents)

	# crop to remove some weird stuff going on at edges.
	singleShape = img_clean[y:y+h, x:x+w]

	# --- Detect Color --- #
	color = findColor(singleShape)

	# print "the intensity is", findShading(singleShape)

	# --- Detect number --- #
	number = len(cnts) #number of contours = number of shapes

	# --- Detect Shading --- #
	shading = findShading(singleShape)


	print "the shape is", shape
	print "the color is", color
	print "the number is ", number 
	print "the shading is", shading, "\n"

	return shape + " " + color + " " + str(number) + " " + shading, img_clean

def findColor(cardImage):

	mean_val = cv2.mean(cardImage)
	height, width, channels = cardImage.shape

	# circleRadius = 200
	# cv2.circle(cardImage, (width/2, height/2), circleRadius, mean_val, -1)

	if (mean_val[0] > mean_val[1] and mean_val[0] > mean_val[2]):
		return "red"
	elif(mean_val[1] > mean_val [0] and mean_val[1] > mean_val[2]):
		return "green"
	else:
		return "blue"

def findShading(cardImage):


	kernel = np.ones((5,5),np.float32)/(25)
	blurred = cv2.filter2D(cardImage,-1,kernel)
	median = cv2.medianBlur(cardImage,101)

	height, width, channels = cardImage.shape
	cardImage2 = cardImage.copy()
	mean_val = cv2.mean(cardImage)
	circleRadius = 200
	cv2.circle(cardImage2, (width/2, height/2), circleRadius, mean_val, -1)

	#take a sample of the center color:
	sample_x_radius = 25
	sample_y_radius = 50
	roi = blurred[height/2 - sample_y_radius:height/2 + sample_y_radius, width/2 - sample_x_radius:width/2 + sample_x_radius]
	mean_center_val = cv2.mean(roi)


	cv2.rectangle(cardImage, (width/2 - sample_x_radius, height/2 - sample_y_radius), (width/2 + sample_x_radius, height/2 + sample_y_radius), mean_center_val, -1)

	# the sum of the mean_center_val is proportional to whiteness.
	sum_mean = sum(mean_center_val) #note that median could do much better here
	print "the sum_mean is", sum_mean

	if (sum_mean > 225*3):
		shading = "blank"
	elif (sum_mean > 400):
		shading = "striped"
	else:
		shading = "filled"


	return shading

	# plt.subplot(2, 2, 1),plt.imshow(cardImage),plt.axis('off'),plt.title('cardImage')
	# plt.subplot(2, 2, 2),plt.imshow(blurred),plt.axis('off'),plt.title('blurred')
	# plt.subplot(2, 2, 3),plt.imshow(median),plt.axis('off'),plt.title('median')
	# plt.subplot(2, 2, 4),plt.imshow(cardImage2),plt.axis('off'),plt.title('average color')

	# plt.show()

def decide_shape_from_extent(extent):
	if (extent < 0.62):
		return("diamond")
	elif(extent >= 0.62 and extent < 0.805):
		return("squiggle")
	else:
		return "oval"

# -----------------------------------------------
# Test Functions                    				
# -----------------------------------------------		
def test_all_fixPerspective(contours, img):
	dst = []
	for c in contours:
		dst.append(fixPerspective(c,img))
	print "Displaying images..."
	displayImages(img, dst)

def test_one_fixPerspective(contours, img, target_contour):
	c = contours[target_contour]
	dst = []
	dst.append(fixPerspective(c,img))
	displayImages(img, dst)

def test_one_identifyFeatures(contours, img, target_contour):
	c = contours[target_contour]
	dst = []
	card_images = []
	dst.append(fixPerspective(c,img))
	title, card_img = identifyFeatures(dst[0])
	card_images.append(card_img)
	displayImages(img, card_images, [title])

def test_all_identifyFeatures(contours, img):
	dst = []
	for c in contours:
		dst.append(fixPerspective(c,img))
	titles = []
	card_images = []
	for d in dst:
		title, card_img = identifyFeatures(d)
		titles.append(title)
		card_images.append(card_img)


	displayImages(img, card_images, titles)

def displayImages(img, dst, additionalTitle = []):
	# assuming in rows of 3
	# probably either 4 or 5 cols.
	num_rows = len(dst) / 3 + 1 #(one more for source)

	plt.subplot(num_rows, 3, 1),plt.imshow(img),plt.title("Original") #display the source at top left
	plt.xticks([])
	plt.yticks([])
	for idx, d in enumerate(dst):
		plt.subplot(num_rows, 3, num_rows*3 - idx)
		if (idx < len(additionalTitle)):
			plt.title(str(idx) + ": " + additionalTitle[idx])
		else:
			plt.title(str(idx))
		plt.xticks([])
		plt.yticks([])
		plt.imshow(d)

	plt.show()

def displayContours(contours, img):
	imgContours = img.copy()
	for c in contours:
		cv2.drawContours(imgContours, c, -1, (0,255,0), 20)
	plt.imshow(imgContours),plt.axis('off'),plt.title("There are " + str(len(contours)) + " contours.")
	plt.show()

def runTests(contours, img):
	print "Running tests given an image and contours."
	print "Select:"
	print "0 - Run fixPerspective on all contours."
	print "1 - Run fixPerspective on one contour."
	print "2 - Run identifyFeatures on one contour."
	print "3 - Run identifyFeatures on all contours."
	print "4 - Display contours"
	selection = int(raw_input())
	if (selection == 0):
		print "Running test_all_fixPerspective."
		test_all_fixPerspective(contours, img)
	elif (selection == 1):
		print "Running test_one_fixPerspective."
		target_contour = int(raw_input("Choose a contour number [0-11]"))
		test_one_fixPerspective(contours, img, target_contour)
	elif (selection == 2):
		print "Running test_one_identifyFeatures." 
		target_contour = int(raw_input("Choose a contour number [0-11]"))
		test_one_identifyFeatures(contours, img, target_contour)
	elif (selection == 3):
		print "Running test_all_identifyFeatures." 
		test_all_identifyFeatures(contours, img)
	elif (selection == 4):
		print "Running displayContours."
		displayContours(contours, img)


	print "All tests finished"

# -----------------------------------------------
# Program begins                       
# -----------------------------------------------

# import image as BGR by default

imageNumber = sys.argv[1]

img = cv2.imread("test_images/set" + imageNumber + ".jpg")

print "creating contours"
contours = createContours(img) # img is the source image numpy array
print "contours have been created. There are", len(contours), "contours."

runTests(contours, img)

