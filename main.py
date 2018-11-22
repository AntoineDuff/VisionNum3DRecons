from camera_calib import calibrate_camera
from read_raw import openraw
import cv2
import numpy as np
import os.path

if __name__ == '__main__':
    
    if (os.path.isfile("cameraD/camera_matrix_d.npy") == False):
        	
        os.mkdir('cameraD')
        (ret_A,
        camera_matrix_A,
        distortion_coefficients_A,
        rotation_vecs_A,
        translation_vecs_A,
        img_points_A,
        object_pts) = calibrate_camera(9, 7, './Calib_d/*.Raw')

        np.save("cameraD/ret_d", ret_A)
        np.save("cameraD/camera_matrix_d",camera_matrix_A)
        np.save("cameraD/distortion_coefficients_d", distortion_coefficients_A)
        np.save("cameraD/rotation_vecs_d", rotation_vecs_A)
        np.save("cameraD/translation_vecs_d", translation_vecs_A)
        np.save("cameraD/img_points_d", img_points_A)
        np.save("cameraD/objects_pts", object_pts)

        print("calibration D done!")

    if (os.path.isfile("cameraG/camera_matrix_g.npy") == False):

        os.mkdir('cameraG')
        (ret_B,
        camera_matrix_B,
        distortion_coefficients_B,
        rotation_vecs_B,
        translation_vecs_B,
        img_points_B,
        object_pts) = calibrate_camera(9, 7, './Calib_g/*.Raw')

        np.save("cameraG/ret_g", ret_B)
        np.save("cameraG/camera_matrix_g",camera_matrix_B)
        np.save("cameraG/distortion_coefficients_g", distortion_coefficients_B)
        np.save("cameraG/rotation_vecs_g", rotation_vecs_B)
        np.save("cameraG/translation_vecs_g", translation_vecs_B)
        np.save("cameraG/img_points_g", img_points_B)
        np.save("cameraG/objects_pts", object_pts)
        print("calibration G done!")

    #matrix D
    obj_pts = np.load("cameraG/objects_pts.npy")
    img_pts_d = np.load("cameraD/img_points_d.npy")
    cam_mat_d = np.load("cameraD/camera_matrix_d.npy")
    dist_cof_d = np.load("cameraA/distortion_coefficients_d.npy")

    #matrix G
    cam_mat_g= np.load("cameraB/camera_matrix_g.npy")
    img_pts_g = np.load("cameraB/img_points_g.npy")
    dist_cof_g = np.load("cameraB/distortion_coefficients_g.npy")

    img = openraw("/Users/antoine/Documents/Vision num/VisionNum3DRecons/Image_d/image_d-11212018153441-0.Raw", 540, 720, 16)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_shape = img.shape[::-1]

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC

    criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(obj_pts,
                                                          img_pts_g,
                                                          img_pts_d,
                                                          cam_mat_g,
                                                          dist_cof_g,
                                                          cam_mat_d,
                                                          dist_cof_d,
                                                          img_shape,
                                                          criteria_stereo,
                                                          flags)

    scale = 0

    RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS,
                                                     dLS,
                                                     MRS,
                                                     dRS,
                                                     img_shape,
                                                     R,
                                                     T,
                                                     scale,
                                                     (0, 0))


    Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                                img_shape, cv2.CV_16SC2)
    Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                                img_shape, cv2.CV_16SC2)

    img = cv2.imread("calibA/a.jpg")
    frameL = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.imread("calibA/b.jpg")

    frameR = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    im1= cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)  # Rectify the image using the calibration parameters founds during the initialisation
    im2= cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    #Draw Red lines
    # for line in range(0, int(im2.shape[0]/20)):
    #     for j in range(0, int(im2.shape[1])): # Draw the Lines on the images Then numer of line is defines by the image Size/20
    #         im2[line*20,j] = 255
    #         im1[line*20,j]= 255

    for line in range(0, int(frameL.shape[0]/20)):
        for j in range(0, int(frameL.shape[1])): # Draw the Lines on the images Then numer of line is defines by the image Size/20
            frameL[line*20,j] = 255
            frameR[line*20,j]= 255

    # im1 = cv2.resize(im1, (600, 300)) 
    # im2 = cv2.resize(im2,(600,300))
    cv2.imshow('Both Images', np.hstack([im1, im2]))
    #cv2.imshow('Normal', np.hstack([frameL, frameR]))

    cv2.waitKey(30000)

