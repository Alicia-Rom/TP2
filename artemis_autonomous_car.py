import imutils
import cv2
import time
import numpy as np
import math, cmath

#Clase principal
class artemis_autonomous_car:
	
	def __init__(self,path, steering_calibration_param=0):

		# Img resolution
		self.img_width = 640
		self.img_height = 480
		self.img_width_center = int(self.img_width/2)
		self.img_height_center = int(self.img_height/2)
		
		# Transformation matrix definition 
		source = np.float32([[0,200],[self.img_width,200],[self.img_width,self.img_height],[0,self.img_height]])
		dest = np.float32([[0,80],[self.img_width,80],[self.img_width_center+90,self.img_height],[self.img_width_center-90,self.img_height]])
		self.transform_matrix=cv2.getPerspectiveTransform(source,dest)

		# Logitudinal control
		self.stop = 0							# At 1 vehicle stoped
		self.throttle_s = 0.04					# Throttle sensitivity
		self.throttle_o = 0.50					# Throttle offset
		self.max_throttle = 0.65				# Maximum throttle
		self.min_longitude_obstacle = 0.4		# Minimum distance to an obstacle
		self.lidar_throttle_control = 0
		self.battery_level = 10

		# Lateral control
		self.k_cross_track_error = 2.5			# Route offset (Stanley modification). It calibrates the point at which the vehicle turns when it sees a curve.
		self.k_Stanley = 1						# Stanley constant (Setting to avoid oscillation)
		self.minimum_contour_area = 350		# Minimum area (in px) for interpreting line point
		self.contour_junction_param =(15,15)	# Parameter for joining close contours and avoiding failure due to noise (Higher number = less noise and less precision).
		self.pixels_in_meter = 1000.0			# Number of pixels of the transformed image equivalents to a meter.
		self.max_larger_width_contour = 200		# Maximum width in pixels of a contour to consider its centre point reliable.
		self.contour_cut_x_position_1 = int(self.img_width/3)	
		self.contour_cut_x_position_2 = int(2*self.img_width/3)
		self.steering_calibration_param = steering_calibration_param
		
		
		# Color definition:
		self.lower_color = (80,21,101)			# Color range 
		self.upper_color = (114,255,255)

		self.frames_counter_switch_fork=15		# Number of frames in which multiple routes must not be observed to consider that a fork has been taken.
		self.path = path						# Vector of numbers defining the route to follow
		self.counter_path = 0
		self.switch_fork=0
		self.counter_switch_fork=self.frames_counter_switch_fork
	
	# Set stop variable
	#	Input:
	#		- stop: 		At 1 vehicle does not move
	
	def set_stop(self,stop):
		self.stop = stop

	# Set battery_level variable
	#	Input:
	#		- battery_level: 		From 0.0 to 10.0. At -1: Battery not connected
	
	def set_battery_level(self, battery_level):
		self.battery_level = battery_level

	# Function that process lidar information [In progress]
	
	def proceso_lidar(self, ranges, show_img):
		min_range=min(ranges[340:360]+ranges[0:20])
		print(min_range)
		throttle_control=self.throttle_s*min_range+self.throttle_o
		if min_range < self.min_longitude_obstacle:
			self.lidar_throttle_control = 0
			print("PARO")
		elif throttle_control > self.max_throttle:
			self.lidar_throttle_control = self.max_throttle
		else:
			self.lidar_throttle_control=throttle_control

		if show_img == True:
			points=[]
			img_lidar = np.zeros((480,640,3), np.uint8)
			img_car = cv2.imread("/home/artemis/Artemis/servicio-servidor/demos/car60.png")
			img_lidar[210:210+img_car.shape[0],307:307+img_car.shape[1]]=img_car
			i=0
			for range in ranges:
				Z=cmath.rect(range,math.radians(-i-90))
				i = i+1
				if Z.real != float('inf') and Z.real != float('-inf'):
					x=int(Z.real*50)+self.img_width_center
				else:
					x=0
				if Z.imag != float('inf') and Z.imag != float('-inf'):
					y=int(Z.imag*50)+self.img_height_center
				else:
					y=0
				cv2.circle(img_lidar,(x,y), radius=2, color=(0,0,255),thickness=2)
			cv2.putText(img_lidar, "Obstacle ahead: "+str(round(min_range*100,3)) + "cm", (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
			cv2.imshow("Mapa LIDAR",img_lidar)
			cv2.waitKey(1)

	
	# Global function that performs all steps of image processing and calculation of control parameters.
	#	Input:
	#		- img: 			Original image
	#		- show_info: 	If true, windows with real-time information are displayed. 
	#		- real_time_control: 0 = (Default) Follow the path set in the path argument, 1 = Take the left route, 2 = Take the middle route, 3 = Take the right route, 4 = Left line change, 5 = Right line change
	#	Output:
	#		- steering_control:	Vehicle steering control parameter.
	#		- throttle_control: Vehicle throttle control parameter.
	#		- trayectory_not_found: At 1 if trayectory not found

	def proceso_fotograma(self, img, show_info, real_time_control=0):
		
		# Perspective transformation
		transformed_img = self.perspective_transformation(img)
		
		# Calculate the heading error and the cross track error 
		(heading_error,cross_track_error,trayectory_not_found,filtered_img,imgLinea)=self.calculate_trajectory(transformed_img,real_time_control,show_info)

		# If trayectory not found
		if trayectory_not_found == 0:
			
			# Implement Stanley Algorithm
			target_angle = self.calculo_stanley(heading_error,cross_track_error,0.52,self.k_Stanley)
			steering_control = self.rad2control(target_angle)
			
			if (self.stop == 1):
				throttle_control=0
			else:
				throttle_control=self.lidar_throttle_control

			if show_info == True:
				cv2.putText(imgLinea, "Heading error: "+str(round(heading_error,3)) + "rad", (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
				cv2.putText(imgLinea, "Cross track error: "+str(round(cross_track_error*100,3)) + "cm", (5,50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
				cv2.putText(imgLinea, "Target steering angle: "+str(round(target_angle,3))+"rad", (5,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
				if real_time_control==0:
					cv2.putText(imgLinea, "Going to: "+str(self.path[self.counter_path]), (5,110),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
				else:
					cv2.putText(imgLinea, "Going to: "+str(real_time_control), (5,110),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
				cv2.putText(img, "Battery level: "+str(self.battery_level), (5,20),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
				cv2.putText(img, "Throttle: "+str(throttle_control), (5,50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
				cv2.imshow("Original frame", img)
				cv2.imshow("Filtred frame", filtered_img)
				cv2.imshow("Transformed frame", imgLinea)
				cv2.waitKey(1)
		
		# If trayectory found:
		elif trayectory_not_found == 1:
			steering_control=0
			throttle_control=0
			if show_info == True:
				cv2.imshow("Original frame", img)
				cv2.imshow("Filtred frame", filtered_img)
				cv2.imshow("Transformed frame", imgLinea)
				cv2.waitKey(1)
		
		return (steering_control,throttle_control,trayectory_not_found)
		
	
	#Function that transforms perspective according to specified transformation matrix
	#	Input:
	#		- img: Original image
	#	Output:
	#		- img: Transformed image

	def perspective_transformation(self,img):
		img=cv2.warpPerspective(img,self.transform_matrix,(self.img_width,self.img_height))
		return img
	
	#Function that calculates the centre of the contours of an image and the number of contours found.
	#	Input:
	#		- img: Filtered image
	#		- offset: height offset in pixels to be added to the calculated points
	#	Output:
	#		- num_contours: Number of contours found.
	#		- centers: Vector of points [[x,y],[x,y],...] indicating the centre of the contours exceeding the margin defined by self.minimum_contour_area
	#		- larger_width_contour: Returns de the width in the X axis of the larger contour
	
	def calculate_center_contours(self,img,offset):
		num_contours=0
		centers=[]
		contours=cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		contours=imutils.grab_contours(contours)
		contours=sorted(contours, key=cv2.contourArea, reverse=True)[:5]
		for contour in contours:
			if num_contours > 2:
				break
			if cv2.contourArea(contour)<self.minimum_contour_area:
				break
			M=cv2.moments(contour)
			center=[int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"])+offset]
			centers.append(center)
			num_contours = num_contours + 1
		centers.sort()
		if num_contours>0:
			larger_width_contour=cv2.boundingRect(contours[0])[2]
		else:
			larger_width_contour=0
		return num_contours,centers,larger_width_contour

	#Function that calculates the angle and deviation of the trajectory from a bird's-eye view image.
	#	Input:
	#		- img: bird's eye view image or front image with perspective transform
	#		- real_time_control: 0 = (Default) Follow the path set in the path argument, 1 = Take the left route, 2 = Take the middle route, 3 = Take the right route, 4 = Take left line for line changing, 5 = Take right line for line changing
	#		- show_info: If true, image with points detected and chosen trayectory is generated
	#	Output:
	#		- heading_error: Angle of the trayectory with respect to the vehicle
	#		- cross_track_error: Transversal deviation in metres of the vehicle with respect to the route
	#		- trayectory_not_found: 0 = Trayectory found, 1 = Trayectory not found
	#		- filtered_img: Filtered image
	#		- img: Input img modified with points detected and chosen trayectory.
	
	def calculate_trajectory(self,img,real_time_control=0,show_info=True):
		
		cropped_img=[]
		num_contours=[0,0,0]
		points=[[],[],[]]
		trayectory_not_found = 0
		heading_error=0
		cross_track_error=0
		point2=[0,0]

		if real_time_control == 0:
			fork=self.path[self.counter_path]
		else:
			fork=real_time_control

		imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)							# From RGB to HSV
		filtered_img = cv2.inRange(imgHSV,self.lower_color,self.upper_color)	# Filter image by color

		# Close contour merging to avoid noise effects
		closing = cv2.morphologyEx(filtered_img,cv2.MORPH_CLOSE,np.ones(self.contour_junction_param,np.uint8))

		# Crop img in sections
		cropped_img.append(closing[380:480, 1:640])
		cropped_img.append(closing[250:350, 1:640])
		cropped_img.append(closing[120:220,1:640])
		
		# First points raw
		num_contours[0],points[0],larger_width_contour=self.calculate_center_contours(cropped_img[0],380)
		if num_contours[0] == 0:	# Zero points detected
			point1=[320,430]			# Take default point
		if num_contours[0] == 1:	# One point detected
			point1=points[0][0]
		if num_contours[0] == 2:	# Two points detected
			if fork == 1:				# Take left path
				point1 = points[0][0]
			if fork == 2:				# Take center path: cetermost point
				if abs(points[0][0][0]-self.img_width_center)<abs(points[0][1][0]-self.img_width_center):
					point1 = points[0][0]
				else:
					point1 = points[0][1]
			if fork == 3:				# Take right path
				point1 = points[0][1]
		if num_contours[0] == 3:	# Three points detected
			point1 = points[0][fork-1]	# Take correponding point
			
		# Second points raw
		num_contours[1],points[1],larger_width_contour=self.calculate_center_contours(cropped_img[1],250)
		if larger_width_contour > self.max_larger_width_contour:	# Width of contour very large: Unreliable recovered points
			# Make 2 cuts to the cropped img for contour splitting and reliable points recovery
			cv2.line(cropped_img[1],(self.contour_cut_x_position_1,0),(self.contour_cut_x_position_1,100),(0,0,0),6,cv2.LINE_AA)
			cv2.line(cropped_img[1],(self.contour_cut_x_position_2,0),(self.contour_cut_x_position_2,100),(0,0,0),6,cv2.LINE_AA)
			num_contours[1],points[1],larger_width_contour=self.calculate_center_contours(cropped_img[1],250)
		print("larger_width_contour = "+str(larger_width_contour))
		if num_contours[1] == 0:		# Zero points detected
			trayectory_not_found=1
		if num_contours[1] == 1:		# One point detected
			point2=points[1][0]
			if real_time_control==0:	
				if self.switch_fork == 1:
					self.counter_switch_fork = self.counter_switch_fork - 1
					if self.counter_switch_fork < 1:
						self.switch_fork = 0
						self.counter_path = self.counter_path + 1
						if self.counter_path == len(self.path)-1:
							self.stop=1
		if num_contours[1] == 2:		# Two points detected
			if fork == 1 or fork == 4:		# Take left path
				point2 = points[1][0]
			if fork == 2:					# Take center path: cetermost point
				if abs(points[1][0][0]-self.img_width_center)<abs(points[1][1][0]-self.img_width_center):
					point2 = points[1][0]
				else:
					point2 = points[1][1]
			if fork == 3 or fork == 5:		# Take right path
				point2 = points[1][1]
			if real_time_control==0:
				self.counter_switch_fork = self.frames_counter_switch_fork
				self.switch_fork = 1
		if num_contours[1] == 3:		# Three points detected
			if fork < 4:				# No lane change
				point2 = points[1][fork-1]	# Take corresponding point
			elif fork == 4:				# Left line change: Take left path
				point2 = points[1][0]
			elif fork == 5:				# Right line change: Take right path
				point2 = points[1][2]
			if real_time_control==0:
				self.counter_switch_fork = self.frames_counter_switch_fork
				self.switch_fork = 1

		# Third points raw
		num_contours[2],points[2],larger_width_contour=self.calculate_center_contours(cropped_img[2],120)
		if (fork > 3) and (num_contours[2] > 1):	# Points only used in case of lane change
			trayectory_not_found=0
			if num_contours[2] == 2:	# Two points detected
				if fork == 4:				# Left lane change: Take left path
					point2 = points[2][0]
				if fork == 5:				# Right lane change: Take right path
					point2 = points[2][1]
			if num_contours[2] == 3:	# Three points detected
				if fork == 4:				# Left lane change: Take left path
					point2 = points[2][0]
				if fork == 5:				# Right lane change: Take right path
					point2 = points[2][2]
			point1=[point2[0],430]		# Replace point1 for point parallel to vehicle axis in function of point2

		# Print data
		print("NUM_CONTOURS: "+str(num_contours))
		print("POINTS: "+str(points))
		print("Trayectory not found: "+ str(trayectory_not_found))
		print("\n")
		#cv2.imshow("Corte 1",cropped_img[0])
		#cv2.imshow("Corte 2",cropped_img[1])
		#cv2.imshow("Corte 3",cropped_img[2])

		# Calculate heading error and cross track error
		if trayectory_not_found == 0:
			heading_error=np.arctan((float(point1[0])-point2[0])/(point1[1]-point2[1]))
			cross_track_error=-(point2[0]-320.0+((point1[0]-point2[0])*self.k_cross_track_error))/self.pixels_in_meter#,point2[1]+(point1[1]-point2[1])*self.k_cross_track_error)/1000.0#-(point1[0]-320.0)/1000.0
		
		# Add info to img
		if show_info == True:
			for points_in_cut in points:
				for point in points_in_cut:
					cv2.circle(img,(point[0],point[1]), radius=2, color=(0,0,255),thickness=5)

			if trayectory_not_found==0:
				cv2.line(img,(point1[0],point1[1]),(point2[0],point2[1]),(0,255,255),6, cv2.LINE_AA)
		
		return (heading_error,cross_track_error,trayectory_not_found,filtered_img,img)

	#Function that converts angle in radians to the vehicle steering control parameter
	#	Input:
	#		- angle_radians: Angle in radians
	#	Output:
	#		- steering_control: Vehicle steering control parameter

	def rad2control(self,angle_radians):
		steering_control=angle_radians/(np.pi/6.0)+self.steering_calibration_param
		if steering_control>1:
			steering_control=1
		if steering_control<-1:
			steering_control=-1
		return steering_control

	#Function that implements the Stanley algorithm
	#	Input:
	#		- heading_error
	#		- cross_track_error
	#		- speed
	#		- k_stanley
	#	Output:
	#		- delta: Steering angle target
	def calculo_stanley(self,heading_error,cross_track_error,speed,k_stanley):
		if speed==0:
			if cross_track_error>0:
				delta=heading_error+np.pi/2.0
			if cross_track_error<0:
				delta=heading_error-np.pi/2.0
			if cross_track_error == 0:
				delta=heading_error
		else:
			delta=heading_error+np.arctan((k_stanley*cross_track_error)/speed)
		return delta
