import cv2
import matplotlib.pyplot as plt
import imutils
import numpy as np

import sys # for testing

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

# should change this to create a seperate image for each card
def createContours(image):


	# preprocess the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

	# find contours in the thresholded image
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
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

#change the image using affine transform such that the contour fits the new image.
#should turn the perspective warped cards into rectangles.
def fixPerspective(contour, image):
	# -----------------------------------------------
	# Fix the perspective.                      
	# -----------------------------------------------
	# get image dimensions
	height, width, channels = image.shape
	
	# set source and dest_rect
	source = findContourCorners(contour, image)

	# dest_rect = tl, tr, bl, br

	dest_rect = [[0,0], [width, 0], [0,height], [width, height]] #may have to make some negative

	pts1 = np.float32(source)
	pts2 = np.float32(dest_rect)

	M = cv2.getPerspectiveTransform(pts1,pts2)
	dst = cv2.warpPerspective(img,M,(width,height))

	# -----------------------------------------------
	# Correct White Balance                       
	# -----------------------------------------------
	img_inv = cv2.bitwise_not(dst)
	white_inv = cv2.threshold(img_inv, 50, 255, cv2.THRESH_TOZERO)[1]
	white = cv2.bitwise_not(white_inv)

	# -----------------------------------------------
	# Display the images.                       
	# -----------------------------------------------
	# plt.subplot(1, 2, 1),plt.imshow(dst),plt.axis('off'),plt.title('dst')
	# plt.subplot(1, 2, 2),plt.imshow(white_inv),plt.axis('off'),plt.title('white')
	# plt.show()

	return white


def identifyFeatures(img):
	cnts = createContours(img)

	height, width, channels = img.shape

	print "height is", height
	print "width is", width

	# img = img[50:height-50, 50:width-50]  # crop the image
	img_inv = cv2.bitwise_not(img)
	# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

	# preprocess the image
	gray = cv2.cvtColor(img_inv, cv2.COLOR_BGR2GRAY)

	thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)[1]

	# find contours in the thresholded image
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1] #version check?

	print "there are ", len(cnts), "contours/shapes!" #number of contours = number of shapes

	#compare the area of the bounding rectangle with the area of the contour
	extents = []
	total_area = 0
	for c in cnts:
		# cv2.drawContours(img, c, -1, (0,255,0), 1)
		area = cv2.contourArea(c)
		total_area += area
		x,y,w,h = cv2.boundingRect(c)
		rect_area = w*h
		extents.append(float(area)/rect_area)
	print "total area is", total_area

	avg_extents = sum(extents)/len(extents)
	shape = decide_shape_from_extent(avg_extents)
	print "the shape is", shape

	#determine the color of the shapes.

	singleShape = img[y:y+h, x:x+w]


	print "the color is", findColor(singleShape)

	# print "the intensity is", findShading(singleShape)



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

	plt.subplot(2, 2, 1),plt.imshow(cardImage),plt.axis('off'),plt.title('cardImage')
	plt.subplot(2, 2, 2),plt.imshow(blurred),plt.axis('off'),plt.title('blurred')
	plt.subplot(2, 2, 3),plt.imshow(median),plt.axis('off'),plt.title('median')
	plt.subplot(2, 2, 4),plt.imshow(cardImage2),plt.axis('off'),plt.title('average color')



	plt.show()


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


def displayImages(img, dst):
	# assuming in rows of 3
	# probably either 4 or 5 cols.
	num_rows = len(dst) / 3 + 1 #(one more for source)

	plt.subplot(num_rows, 3, 1),plt.imshow(img),plt.axis('off') #display the source at top left

	for idx, d in enumerate(dst):
		plt.subplot(num_rows, 3, num_rows*3 - idx),plt.imshow(d),plt.axis('off')

	plt.show()

def runTests(contours, img):
	print "Running tests given an image and contours."
	print "Select:"
	print "0 - Run fixPerspective on all contours."
	print "1 - Run fixPerspective on one contour."
	selection = int(raw_input())
	if (selection == 0):
		print "Running test_all_fixPerspective."
		test_all_fixPerspective(contours, img)
	if (selection == 1):
		print "Running test_one_fixPerspective."
		target_contour = int(raw_input("Choose a contour number [0-11]"))
		test_one_fixPerspective(contours, img, target_contour)

	print "All tests finished"





# -----------------------------------------------
# Program begins                       
# -----------------------------------------------

# import image
img = cv2.imread("test_images/set2.jpg")
contours = createContours(img) # img is the source image numpy array

runTests(contours, img)




# identifyFeatures(dst[int(sys.argv[1])])

# print "Colors:"

# for d in dst:
# 	print findColor(d)



# # Display the images: ------------------
# # assuming in rows of 3
# # probably either 4 or 5 cols.
# num_rows = len(contours) / 3 + 1 #(one more for source)

# plt.subplot(num_rows, 3, 1),plt.imshow(img),plt.axis('off') #display the source at top left

# for idx, d in enumerate(dst):
# 	plt.subplot(num_rows, 3, num_rows*3 - idx),plt.imshow(d),plt.axis('off')

# plt.show()
# End display the images ----------------

