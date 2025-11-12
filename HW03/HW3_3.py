import cv2
import numpy as np
from pupil_apriltags import Detector
import gtsam
from gtsam import Pose3, Point3, Rot3, Cal3_S2
from gtsam import GenericProjectionFactorCal3_S2, NonlinearFactorGraph
from gtsam import PriorFactorPoint3, LevenbergMarquardtOptimizer, Values
from gtsam import BetweenFactorPose3, PriorFactorPose3
import glob #importing to sort through all 500 datafiles without manuually entering their directory names. 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

camera_mat = [a_u, a_v, p_u, p_v] # reformatted as array to fit the apriltag detect function. 
K = Cal3_S2(a_u, a_v, 0.0, p_u, p_v)

tag_side = 0.01

at_detector = Detector(families="tag36h11") # setting apriltag family

# importing images
vslam_images =glob.glob("/Users/yash/Documents/Python Code/vslam/frame_*.jpg")

# Detect tags in all images
all_detections = [] # creating a blank list of all tags detected
all_tags = set()

#detecting every tag in every photo while also estimating pose
for i, img_path in enumerate(vslam_images):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    tags = at_detector.detect(img, estimate_tag_pose=True, camera_params=camera_mat,tag_size=tag_side)
    
    tag_detections = [] # empty list of all detected tags in image
    for tag in tags: # looping through ALL tags in given camera pose 
        if tag.pose_R is not None and tag.pose_t is not None:
            tag_detections.append({
                'tag_id': tag.tag_id,
                'R': tag.pose_R,
                't': tag.pose_t.flatten() # to maintain comaptibility with GTSam input argument format
            })
            all_tags.add(tag.tag_id) # tracking all unique tags in all images. 
    
    all_detections.append(tag_detections) #creating a master list of ALL detections in every camera frame/pose(Image)

print(f"Image processing for tag detection complete")

# Factor Graph Calculation 

# Build factor graph for SLAM
graph = NonlinearFactorGraph()

# Add between factors for each tag detection
noise = gtsam.noiseModel.Isotropic.Sigma(6, 1)

# creating factor graph based on camera poses X from i=1 to i=500, 

for i in range(len(vslam_images)):
    for detection in all_detections[i]:
        tag_id = detection['tag_id']
        
        # Create measured pose from tag to camera
        R = Rot3(detection['R'])
        t = Point3(detection['t'][0], detection['t'][1], detection['t'][2])
        measured_pose = Pose3(R, t) # extracting pose for every tag ID in every image. 

        # Create keys
        camera_key = gtsam.symbol('x', i)
        tag_key = gtsam.symbol('y', tag_id)

    
        
        # Add between factor
        factor = BetweenFactorPose3(camera_key, tag_key, measured_pose, noise)
        graph.add(factor)

# Fix first tag detected at origin. Need to set the prior factor pose3 for tag0

#extracting tagID of first tag from first image
first_tag_id = None
first_tag_id = all_detections[0][0]['tag_id']

tag0 = gtsam.symbol('y', first_tag_id)
origin = Pose3(Rot3(), Point3(0, 0, 0))
fixed_noise = gtsam.noiseModel.Constrained.All(6)
graph.add(PriorFactorPose3(tag0, origin, fixed_noise))

print("First Tag ID assigned to the orgin is: ",first_tag_id)

# Initialize pose values
initial = Values()

# Initialize camera poses
for i in range(len(all_detections)):
    camera_key = gtsam.symbol('x', i)
    initial_pose = Pose3(Rot3(), Point3(0, 0, 0))
    initial.insert(camera_key, initial_pose)

# Setting first occurence of every tag to be its initial pose. 

print(all_tags)

for tag_id in all_tags: # looping through all unique tags known 
    for i, detections in enumerate(all_detections): # looping through all indices and tags detected in ALL images
        for d in detections: # looping throgh all detections in a particular image
            if d['tag_id'] == tag_id: # comparing every tag to the parent list of all tags from line 159
                R = Rot3(d['R'])
                t = Point3(d['t'][0], d['t'][1], d['t'][2])
                tag_pose = Pose3(R, t)
                
                tag_key = gtsam.symbol('y', tag_id) # setting THAT image, THAT detectionm for THAT tag as initial world pose in factor graph
                initial.insert(tag_key, tag_pose)
                break # once assiged, end loop. 
        else:
            continue
        break

# Optimize
print("\nOptimizing...")
optimizer = LevenbergMarquardtOptimizer(graph, initial)
result = optimizer.optimize()
print("Optimization done!")

# Extract camera and tag positions for plotting
camera_positions = []
tag_positions = []
tag_ids = []

for i in range(len(vslam_images)):
    camera_key = gtsam.symbol('x', i)
    pose = result.atPose3(camera_key)
    t = pose.translation()
    camera_positions.append([t[0], t[1], t[2]])

for tag_id in sorted(all_tags):
    tag_key = gtsam.symbol('y', tag_id)
    pose = result.atPose3(tag_key)
    t = pose.translation()
    tag_positions.append([t[0], t[1], t[2]])
    tag_ids.append(tag_id)

camera_positions = np.array(camera_positions)
tag_positions = np.array(tag_positions)

# Plot the results
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(projection='3d')

ax.scatter(tag_positions[:, 0], tag_positions[:, 1], tag_positions[:, 2], 
           c='red', marker='s', s=100, label='AprilTags', alpha=0.8)

for i, tag_id in enumerate(tag_ids):
    ax.text(tag_positions[i, 0], tag_positions[i, 1], tag_positions[i, 2], f'  {tag_id}', fontsize=8)

ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], c='blue', marker='o', s=100, label='Camera Poses', alpha=0.8)

# Labels and title
ax.set_xlabel('X (meters)', fontsize=12)
ax.set_ylabel('Y (meters)', fontsize=12)
ax.set_zlabel('Z (meters)', fontsize=12)
ax.set_title('Estimated Camera Poses and AprilTag Positions', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.show()

