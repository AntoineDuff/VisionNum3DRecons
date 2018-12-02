import numpy as np
import cv2
import glob
# import matplotlib.pyplot as plt
# import math


def calibrate_camera(i_grid_size, j_grid_size, file_path):

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Values of the object points and image points from all the images.
    objp = np.zeros((i_grid_size * j_grid_size, 3), np.float32)
    objp[:,:2] = np.mgrid[0:i_grid_size, 0:j_grid_size].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # Importation of the images.
    images = glob.glob(str(file_path)+'/*.Raw')
    images.sort()


    for image in images:

        img = openraw(image, 540, 720, 16)
        img = np.uint8(img * 255)
        gray = img

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (i_grid_size, j_grid_size), None)

        # If found, add object points, image points (after refining them), if not, print name of image
        if ret == False:
            print(image)

        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, (i_grid_size, j_grid_size), corners2, ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    # Calibration of one of the camera.
    (ret,
    camera_matrix,
    distortion_coefficients,
    rotation_vecs,
    translation_vecs) = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None,  None)

    return ret, camera_matrix, distortion_coefficients, rotation_vecs, translation_vecs, imgpoints, objpoints


def calibrate_stereo(i_grid_size, j_grid_size, file_path):
    
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Values of the object points and image points from all the images.
    objp = np.zeros((i_grid_size * j_grid_size, 3), np.float32)
    objp[:,:2] = np.mgrid[0:i_grid_size, 0:j_grid_size].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # Importation of the images.
    images = glob.glob(str(file_path)+'/*.Raw')
    images.sort()

    for image in images:

        img = openraw(image, 540, 720, 16)
        img = np.uint8(img * 255)
        gray = img

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (i_grid_size, j_grid_size), None)

        # If found, add object points, image points (after refining them), if not, print name of image
        if ret == False: 
            print(image)

        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, (i_grid_size, j_grid_size), corners2, ret)
            # cv2.imshow(str(image),img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    return objpoints, imgpoints
    

def rectify_camera(obj_pts, img_pts_A, img_pts_B, cam_mat_A, cam_mat_B, dist_cof_A, dist_cof_B, new_cam_A, new_cam_B):

    img_shape = (540, 720)

    flags = cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_USE_INTRINSIC_GUESS
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(obj_pts,
                                                          img_pts_B,
                                                          img_pts_A,
                                                          cam_mat_B,
                                                          dist_cof_B,
                                                          cam_mat_A,
                                                          dist_cof_A,
                                                          img_shape,
                                                          criteria_stereo,
                                                          flags=flags)

    RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(cam_mat_B,
                                                      dLS,
                                                      cam_mat_A,
                                                      dRS,
                                                      (800, 800),
                                                      R,
                                                      T,
                                                      alpha=-1, flags=0)

    Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                                  (800, 800), cv2.CV_16SC2)

    Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                                   (800, 800), cv2.CV_16SC2)

    return Left_Stereo_Map, Right_Stereo_Map


def openraw(namefile, width, heigth, bit):

    #Ouverture des fichiers d'image
    with open(namefile,'r', encoding='utf-8', errors='ignore') as fdata:

        signal = np.fromfile(fdata, np.uint16).reshape((width,heigth))

    signal = signal * np.ones([width, heigth])/(2**bit)
    
    return signal


def detectpoint(image, threshold):
    ret, thresh = cv2.threshold(image,threshold, 255 ,0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    #plt.imshow(im2)

    #Take longuest contour
    cnt = max(contours, key=len)


    M = cv2.moments(cnt)


    #centroid

    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    print(cx, cy)

    return 0



#img = openraw("Image_g/image_g-11212018153638-0.raw", 540, 720, 16)
#print(img)
#img = np.uint8(img * 255)
#plt.imshow(image_8bit)
#cv2.imread(img)
#print(img.dtype)
#plt.imshow(img)
#detectpoint(image_8bit, 200)
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#plt.show()











