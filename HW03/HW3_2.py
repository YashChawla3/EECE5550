import cv2
import numpy as np
from pupil_apriltags import Detector
import gtsam
from gtsam import Pose3
from gtsam import Point3
from gtsam import Point2
from gtsam import Rot3
from gtsam import Cal3_S2, GenericProjectionFactorCal3_S2, NonlinearFactorGraph
from gtsam import PriorFactorPoint3, PriorFactorPoint2
from gtsam import LevenbergMarquardtOptimizer
from gtsam import Values
from gtsam.symbol_shorthand import X, L


#Create the 3D coordinates of the checkerboard corners
# This represents where the corners are in the real world (assuming flat board at z=0)

objp = np.zeros((48, 3), np.float32) #creating 3D point referneces that are passed to every image as a 
#refernece to correlate coorindates between returned real world points. 

# Fill in the coordinates using loops
index = 0
for y in range(6):  # 6 rows
    for x in range(8):  # 8 columns
        objp[index] = [x, y, 0]
        index += 1
side_length = 0.01
objp = objp*side_length # scaling side length to be 0.01m

#Create empty lists to store points
objpoints = []  # 3D points references from the checkerboard that is passed objp array. 
imgpoints = []  # 2D point correspondences from the images

# List of image files
image_paths = [
    "/Users/yash/Documents/Python Code/calibration_images/IMG_3910.JPEG",
    "/Users/yash/Documents/Python Code/calibration_images/IMG_3913.JPEG",
    "/Users/yash/Documents/Python Code/calibration_images/IMG_3914.JPEG",
    "/Users/yash/Documents/Python Code/calibration_images/IMG_3915.JPEG",
    "/Users/yash/Documents/Python Code/calibration_images/IMG_3916.JPEG",
    "/Users/yash/Documents/Python Code/calibration_images/IMG_3917.JPEG",
    "/Users/yash/Documents/Python Code/calibration_images/IMG_3918.JPEG",
    "/Users/yash/Documents/Python Code/calibration_images/IMG_3919.JPEG",
]

# Process each image
for image_path in image_paths:
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find checkerboard corners
    found, corners = cv2.findChessboardCorners(gray, (8, 6), None)
  
    if found:
        #print(f"Corners found in {image_path}")
        # Add the 3D points and 2D points to our lists
        objpoints.append(objp) # adding corner index to the obj points list 
        imgpoints.append(corners) # adding corner real world coordinates to img points list. 
    else:
        
        #print(f"Corners not found in {image_path}")
        continue

# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#print(f"Camera matrix:\n{mtx}")

# Start of section 2 to get factor graphs running

# Creating Camera Matrix as SE3 Matrix 

a_u = mtx[0][0]
a_v = mtx[1][1]
p_u = mtx[0][2]
p_v = mtx[1][2]

K = Cal3_S2(a_u, a_v, 0.0, p_u, p_v)

at_detector = Detector(families="tag36h11") # setting apriltag family

img = cv2.imread("/Users/yash/Documents/Python Code/vslam/frame_0.jpg", cv2.IMREAD_GRAYSCALE) # importing image
tags = at_detector.detect(img)
tag0 = tags[0]
print(tag0)

noise = gtsam.noiseModel.Isotropic.Sigma(2, 0.01)

corners_2 = tag0.corners # 2D Corner coordinates from the camera
print(corners_2)

tag_size = 0.01

#3D Tag Corners based on known geometry information of tagh 
corners_3 = np.array([[-tag_size/2, tag_size/2, 0],
                     [tag_size/2, tag_size/2, 0 ],
                     [tag_size/2, -tag_size/2, 0],
                     [-tag_size/2, -tag_size/2, 0]
                    ])


graph = NonlinearFactorGraph() # creating a blank factor graph

#adding tag information as a factor to the graph
for i in range(4):
    point_2d = gtsam.Point2(corners_2[i, 0], corners_2[i, 1]) # extracting 2d points from apriltag
    factor = gtsam.GenericProjectionFactorCal3_S2(point_2d, noise, X(0), L(i), K) #adding to factor graph with some pixel noise, for camera pose X(0) as landmark (1)
    graph.add(factor)

#adding landmark information as a prior factor
fixed_noise = gtsam.noiseModel.Constrained.All(3)

for i in range(4):
    point_3d = gtsam.Point3(corners_3[i,0],corners_3[i,1], corners_3[i,2])
    prior = PriorFactorPoint3(L(i),point_3d, fixed_noise)
    graph.add(prior)

initial_guess = Values()
initial_pose = Pose3(Rot3(), Point3(0.0,0.0,-0.05)) # <- Initial estimate pose

initial_guess.insert(X(0), initial_pose)

for i in range(4):
    initial_guess.insert(L(i), Point3(corners_3[i, 0], corners_3[i, 1], corners_3[i, 2]))

optimized = LevenbergMarquardtOptimizer(graph, initial_guess)
end_rotation = optimized.optimize()

print(" Optimized!")

camera_pose = end_rotation.atPose3(X(0))
print(camera_pose)



