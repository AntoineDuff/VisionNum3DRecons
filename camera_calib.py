import numpy as np
import cv2
import glob

def calibrate_camera(i_grid_size, j_grid_size):

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((i_grid_size * j_grid_size, 3), np.float32)
    objp[:,:2] = np.mgrid[0:i_grid_size, 0:j_grid_size].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('./calibration/*.jpg')

    for image in images:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (i_grid_size, j_grid_size), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (i_grid_size, j_grid_size), corners2, ret)

    img = cv2.imread(images[0], 0)
    (ret,
    camera_matrix,
    distortion_coefficients,
    rotation_vecs,
    translation_vecs) = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
    
    return ret, camera_matrix, distortion_coefficients, rotation_vecs, translation_vecs

