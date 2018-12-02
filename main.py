from camera_calib import calibrate_camera, rectify_camera, openraw, calibrate_stereo
import cv2
import numpy as np
import os.path
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    if (os.path.isfile("cameraA/camera_matrix_A.npy") == False):
        	
        os.mkdir('cameraA')
        (ret_A,
        camera_matrix_A,
        distortion_coefficients_A,
        rotation_vecs_A,
        translation_vecs_A,
        img_points_A,
        object_pts) = calibrate_camera(15, 10, './calib_camera_d')

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
        object_pts) = calibrate_camera(15, 10, './calib_camera_g')

        np.save("cameraB/ret_B", ret_B)
        np.save("cameraB/camera_matrix_B",camera_matrix_B)
        np.save("cameraB/distortion_coefficients_B", distortion_coefficients_B)
        np.save("cameraB/rotation_vecs_B", rotation_vecs_B)
        np.save("cameraB/translation_vecs_B", translation_vecs_B)
        np.save("cameraB/img_points_B", img_points_B)
        np.save("cameraB/objects_pts", object_pts)
        print("calibration B done!")

    #matrix A
    #obj_pts = np.load("cameraB/objects_pts.npy")
    #img_pts_A = np.load("cameraA/img_points_A.npy")
    cam_mat_A = np.load("cameraA/camera_matrix_A.npy")
    dist_cof_A = np.load("cameraA/distortion_coefficients_A.npy")
    #rot_vecs_A = np.load("cameraA/rotation_vecs_A.npy")


    #matrix B
    cam_mat_B= np.load("cameraB/camera_matrix_B.npy")
    #img_pts_B = np.load("cameraB/img_points_B.npy")
    dist_cof_B = np.load("cameraB/distortion_coefficients_B.npy")
    #rot_vecs_B = np.load("cameraB/rotation_vecs_B.npy")

    
   
    new_cam_A, roiA = cv2.getOptimalNewCameraMatrix(cam_mat_A, dist_cof_A, (540, 720), 1, (540, 720))
    new_cam_B, roiB = cv2.getOptimalNewCameraMatrix(cam_mat_B, dist_cof_B, (540, 720), 1, (540, 720))


    frameL = openraw("image_khan_g/image_g-11292018094748-0.Raw", 540, 720, 16)
    frameL = np.uint8(frameL * 255)

    frameR = openraw("image_khan_d/image_d-11292018094727-0.Raw", 540, 720, 16)
    frameR = np.uint8(frameR * 255)


    obj_pts, img_pts_A = calibrate_stereo(15, 10, "calib_stereo_d")
    obj_pts, img_pts_B = calibrate_stereo(15, 10, "calib_stereo_g")



    Left_Stereo_Map, Right_Stereo_Map = rectify_camera(obj_pts, img_pts_A, img_pts_B, cam_mat_A, cam_mat_B, dist_cof_A, dist_cof_B, new_cam_A, new_cam_B)

    Left_nice= cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)  # Rectify the image using the calibration parameters founds during the initialisation
    Right_nice= cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    #print(Right_nice)
    plt.imshow(Left_nice, cmap = "Greys")
    plt.show()
    plt.imshow(Right_nice, cmap = "Greys")
    plt.show()


