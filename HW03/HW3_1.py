import cv2
import numpy as np

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
        # Add the 3D points and 2D points to our lists
        objpoints.append(objp) # adding corner index to the obj points list 
        imgpoints.append(corners) # adding corner real world coordinates to img points list. 
    else:
        continue

# Calibrate the camera
ret_type, matrix, distortion, rot, trans = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(f"Camera matrix:\n{matrix}")


