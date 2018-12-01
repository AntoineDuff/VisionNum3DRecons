from camera_calib import calibrate_camera
import cv2
import numpy as np
import os.path

if __name__ == '__main__':
    
    if (os.path.isfile("cameraA/camera_matrix_A.npy") == False):
        	
        os.mkdir('cameraA')
        (ret_A,
        camera_matrix_A,
        distortion_coefficients_A,
        rotation_vecs_A,
        translation_vecs_A,
        img_points_A,
        object_pts) = calibrate_camera(6, 9, './calibA/a.jpg')

        np.save("cameraA/ret_A", ret_A)
        np.save("cameraA/camera_matrix_A",camera_matrix_A)
        np.save("cameraA/distortion_coefficients_A", distortion_coefficients_A)
        np.save("cameraA/rotation_vecs_A", rotation_vecs_A)
        np.save("cameraA/translation_vecs_A", translation_vecs_A)
        np.save("cameraA/img_points_A", img_points_A)
        np.save("cameraA/objects_pts", object_pts)

        print("calibration A done!")

    if (os.path.isfile("cameraB/camera_matrix_B.npy") == False):

        os.mkdir('cameraB')
        (ret_B,
        camera_matrix_B,
        distortion_coefficients_B,
        rotation_vecs_B,
        translation_vecs_B,
        img_points_B,
        object_pts) = calibrate_camera(6, 9, './calibA/b.jpg')

        np.save("cameraB/ret_B", ret_A)
        np.save("cameraB/camera_matrix_B",camera_matrix_A)
        np.save("cameraB/distortion_coefficients_B", distortion_coefficients_A)
        np.save("cameraB/rotation_vecs_B", rotation_vecs_A)
        np.save("cameraB/translation_vecs_B", translation_vecs_A)
        np.save("cameraB/img_points_B", img_points_B)
        np.save("cameraB/objects_pts", object_pts)
        print("calibration B done!")

    #matrix A
    obj_pts = np.load("cameraB/objects_pts.npy")
    img_pts_A = np.load("cameraA/img_points_A.npy")
    cam_mat_A = np.load("cameraA/camera_matrix_A.npy")
    dist_cof_A = np.load("cameraA/distortion_coefficients_A.npy")

    #matrix B
    cam_mat_B= np.load("cameraB/camera_matrix_B.npy")
    img_pts_B = np.load("cameraB/img_points_B.npy")
    dist_cof_B = np.load("cameraB/distortion_coefficients_B.npy")

    img = cv2.imread("calibA/a.jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_shape = gray.shape[::-1]

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC

    criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(obj_pts,
                                                          img_pts_B,
                                                          img_pts_A,
                                                          cam_mat_B,
                                                          dist_cof_B,
                                                          cam_mat_A,
                                                          dist_cof_A,
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

    Left_nice= cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)  # Rectify the image using the kalibration parameters founds during the initialisation
    Right_nice= cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    print("ok")

    # (ret_B,
    # camera_matrix_B,
    # distortion_coefficients_B,
    # rotation_vecs_B,
    # translation_vecs_B) = calibrate_camera(6, 9, './calibB/*.jpg')

    # cam = cv2.VideoCapture(0)

    # while True:
    #     ret, frame = cam.read()
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     cv2.imshow('frame', gray)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

# When everything done, release the capture
# cam.release()
# cv2.destroyAllWindows()

