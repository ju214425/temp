import cv2 as cv
import math
import numpy as np
import os

# input_img_path = "/Users/leejonguk/Desktop/hanyang/project/iris/MMU2_Iris_Database/MMU2_Iris_Output/31/01/310103.bmp"
input_img_path = "/Users/leejonguk/Desktop/hanyang/project/iris/CASIA-IrisV4(JPG)/CASIA-Iris-Interval/135/R/S1135R02.jpg"

def adjust_gamma(image, gamma=1.0):
  
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
	for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(image, table)


def pre_processing(image):
	cimg = image
    
	kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
	blackhat = cv.morphologyEx(cimg, cv.MORPH_BLACKHAT, kernel)
	return cv.add(blackhat, cimg)

def ROI_detection(image):
	cimg = image

	#output = (input, kernel_size)
	cimg = cv.medianBlur(cimg, 17)
	# output = (input, kernel_size, sigma_X, sigma_Y)
	cimg = cv.GaussianBlur(cimg, (13, 13), 2) 
	# output = (input, thershold1, thershold2)
	# cimg = cv.Canny(cimg, 10, 20)

	cimg = adjust_gamma(cimg, 10)


	return cimg

def hough_Circle(image):
	cimg = image
	cimg = pre_processing(cimg)
	cimg = ROI_detection(cimg)
	circles = cv.HoughCircles(cimg, cv.HOUGH_GRADIENT, 1, 20, param1=20, 
								param2=20, minRadius=0, maxRadius=100)

	if circles is not None:
		inner_circle = np.uint16(np.around(circles[0][0])).tolist()
	else:
		print("inner_cricles not found")
	iris_frame = cv.circle(image, (inner_circle[0], inner_circle[1]), inner_circle[2], (0,0,0), cv.FILLED)

	size = int(inner_circle[2] * 2.4)

	return_frame = iris_frame[(inner_circle[1] - size):
					(inner_circle[1] + size),
					(inner_circle[0] - size):
					(inner_circle[0] + size)]

	circles = cv.HoughCircles(cimg, cv.HOUGH_GRADIENT, 1, 20, param1=20, 
								param2=20, minRadius=inner_circle[2]+5, maxRadius = 150)

	outer_circle = [0, 0, 0]
	if circles is not None:
		print(circles)
		outer_circle = np.uint16(np.around(circles[0][0])).tolist()
	else:
		print("outer_circles not found")
		outer_circle = inner_circle
		outer_circle[2] = int(inner_circle[2] * 2.4)


	print(inner_circle)
	print(outer_circle)
	# iris_frame = cv.circle(image, (outer_circle[0], outer_circle[1]), outer_circle[2], (0,0,0), cv.LINE_4)

	mask = cv.bitwise_not(
				cv.circle(np.zeros((np.size(frame,0),np.size(frame,1),1), np.uint8)
					, (outer_circle[0], outer_circle[1]), outer_circle[2], (255,255,255), cv.FILLED))
	iris_frame = frame.copy()
	iris_frame = cv.subtract(frame, frame, iris_frame, mask, -1)
	

	# return return_frame
	return iris_frame[(outer_circle[1] - outer_circle[2]):
						(outer_circle[1] + outer_circle[2]),
						(outer_circle[0] - outer_circle[2]):
						(outer_circle[0] + outer_circle[2])], outer_circle[2]

def getPolar2CartImg(image, rad):
	
	c = (float(np.size(image, 0)/2.0), float(np.size(image, 1)/2.0))
	
	imgRes = cv.warpPolar(image, (rad*3,360), c, np.size(image, 0)/2, cv.WARP_POLAR_LOG)
	
	for valid_width in reversed(range(rad*3)):
		blank_cnt = 0
		for h in range(360):
			if (imgRes[h][valid_width] != 0):
				blank_cnt+=1
		if(blank_cnt == 0):

			
			imgRes = imgRes[0:360, valid_width:rad*3]
			break

	imgRes = cv.resize(imgRes, (80, 360), interpolation=cv.INTER_CUBIC)
	

	return (imgRes)


frame = cv.imread(input_img_path, cv.CV_8UC1)
cv.imshow("input", frame)

# bottom_hat_filtered = pre_processing(frame)
# cv.imshow("bottom_hat_filtered", bottom_hat_filtered)

# ROI_img = ROI_detection(bottom_hat_filtered)
# cv.imshow("ROI_img", ROI_img)

inner_circles, radius = hough_Circle(frame)
# temp = cv.add(frame, circles)
cv.imshow("inner_circles", inner_circles)

norm_image = getPolar2CartImg(inner_circles, radius)
cv.imshow("norm_image", norm_image)
# outer_circles = hough_Circle_Outer(inner_circles)
# cv.imshow("outer_circles", outer_circles)




# cv.imshow("temp", temp)

cv.waitKey(100000)