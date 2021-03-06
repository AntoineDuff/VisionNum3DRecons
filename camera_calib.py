import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


def calibrate_camera(i_grid_size, j_grid_size, file_path):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Values of the object points and image points from all the images.
    objp = np.zeros((i_grid_size * j_grid_size, 3), np.float32)
    objp[:, :2] = np.mgrid[0:i_grid_size, 0:j_grid_size].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # Importation of the images.
    images = glob.glob(str(file_path) + '/*.Raw')
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
     translation_vecs) = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)

    return ret, camera_matrix, distortion_coefficients, rotation_vecs, translation_vecs, imgpoints, objpoints


def calibrate_stereo(i_grid_size, j_grid_size, file_path_A, file_path_B):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Values of the object points and image points from all the images.
    objp = np.zeros((i_grid_size * j_grid_size, 3), np.float32)
    objp[:, :2] = np.mgrid[0:i_grid_size, 0:j_grid_size].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpointsA = []  # 2d points in image plane.
    imgpointsB = []  # 2d points in image plane.

    # Importation of the images.
    images_A = glob.glob(str(file_path_A) + '/*.Raw')
    images_B = glob.glob(str(file_path_B) + '/*.Raw')
    images_A.sort()
    images_B.sort()

    for (ima, imb) in zip(images_A, images_B):

        imga = openraw(ima, 540, 720, 16)
        imga = np.uint8(imga * 255)
        graya = imga

        imgb = openraw(imb, 540, 720, 16)
        imgb = np.uint8(imgb * 255)
        grayb = imgb

        # Find the chess board corners
        reta, cornersa = cv2.findChessboardCorners(graya, (i_grid_size, j_grid_size), None)
        retb, cornersb = cv2.findChessboardCorners(grayb, (i_grid_size, j_grid_size), None)

        # If found, add object points, image points (after refining them), if not, print name of image
        if reta == False or retb == False:
            print(ima, imb)

        if reta == True and retb == True:
            objpoints.append(objp)

            corners2a = cv2.cornerSubPix(graya, cornersa, (11, 11), (-1, -1), criteria)
            corners2b = cv2.cornerSubPix(grayb, cornersb, (11, 11), (-1, -1), criteria)
            imgpointsA.append(corners2a)
            imgpointsB.append(corners2b)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(imga, (i_grid_size, j_grid_size), cornersa, reta)
            cv2.imshow(str(ima), img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            img = cv2.drawChessboardCorners(imgb, (i_grid_size, j_grid_size), cornersb, retb)
            cv2.imshow(str(imb), img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return objpoints, imgpointsA, imgpointsB


def rectify_camera(obj_pts, img_pts_A, img_pts_B, cam_mat_A, cam_mat_B, dist_cof_A, dist_cof_B):
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

    print(R)
    print(cv2.Rodrigues(R))
    print(T)
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

    return Left_Stereo_Map, Right_Stereo_Map, PL, PR


def openraw(namefile, width, heigth, bit):
    # Ouverture des fichiers d'image
    with open(namefile, 'r', encoding='utf-8', errors='ignore') as fdata:
        signal = np.fromfile(fdata, np.uint16).reshape((width, heigth))

    signal = signal * np.ones([width, heigth]) / (2 ** bit)

    return signal


def detectpoint(image, bg, threshold, d):
    img = openraw(str(image), 540, 720, 8)
    # bg = openraw(str(bg), 540, 720, 8)

    # Load parameter of stereo pair
    Left_Stereo_Map = [np.load("camera_stereo/Left_Stereo_Map0.npy"), np.load("camera_stereo/Left_Stereo_Map1.npy")]
    Right_Stereo_Map = [np.load("camera_stereo/Right_Stereo_Map0.npy"), np.load("camera_stereo/Right_Stereo_Map1.npy")]

    # Rectify the image using the calibration parameters founds during the initialisation
    if d == False:
        image_corr = cv2.remap(img, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    else:
        image_corr = cv2.remap(img, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    ret, thresh = cv2.threshold(image_corr, threshold, 255, 0)

    thresh = np.uint8(thresh)
    # plt.imshow(thresh)
    # plt.show()
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    out = 0
    M = 0

    if len(contours) == 0:
        out = 1
    else:
        # Take longuest contour
        cnt = max(contours, key=len)
        M = cv2.moments(cnt)

    # centroid
    if out == 1 or M == 0 or M['m00'] == 0:
        pts = np.array([0, 0], dtype=np.float)
    else:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        pts = np.array([cx, cy], dtype=np.float)

    return pts


def triangulation(PR, PL, coor_pts_R, coor_pts_L):
    # Homogeneous triangulation and return to cartesian coordinate
    pst_triang = cv2.triangulatePoints(PR, PL, coor_pts_R, coor_pts_L)
    pst_triang /= pst_triang[3]

    return pst_triang

def triangulation2(PR, PL, coor_pts_R, coor_pts_L):
    # Homogeneous triangulation and return to cartesian coordinate
    pst_triang = cv2.triangulatePoints(PR, PL, coor_pts_R, coor_pts_L)

    return pst_triang

def pts_detection(filepath_d, filepath_g):
    # Importation of the images.
    images_d = glob.glob(str(filepath_d) + '/*.Raw')
    images_g = glob.glob(str(filepath_g) + '/*.Raw')
    images_d.sort()
    images_g.sort()

    pts_3d_d = []
    pts_3d_g = []

    for (imd, img) in zip(images_d, images_g):
        pts_d = detectpoint(imd, 0, 100, d=True)
        pts_g = detectpoint(img, 0, 100, d=False)

        pts_3d_d.append(pts_d)
        pts_3d_g.append(pts_g)

    return pts_3d_d, pts_3d_g
