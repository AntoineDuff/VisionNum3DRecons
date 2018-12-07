from camera_calib import calibrate_camera, rectify_camera, openraw, calibrate_stereo, triangulation
import cv2
import numpy as np
import os.path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


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

    # Load parameter of matrix A -> d
    cam_mat_A = np.load("cameraA/camera_matrix_A.npy")
    dist_cof_A = np.load("cameraA/distortion_coefficients_A.npy")

    # Load parameter of matrix B  -> g
    cam_mat_B = np.load("cameraB/camera_matrix_B.npy")
    dist_cof_B = np.load("cameraB/distortion_coefficients_B.npy")

    if (os.path.isfile("camera_stereo/img_points_stereo_B.npy") == False):

        os.mkdir('camera_stereo')

        obj_pts, img_pts_A, img_pts_B = calibrate_stereo(15, 10, "calib_stereo_d", "calib_stereo_g")
        #obj_pts_B, img_pts_B = calibrate_stereo(15, 10, "calib_stereo_g")

        (Left_Stereo_Map, Right_Stereo_Map, PL, PR) = rectify_camera(obj_pts, img_pts_A, img_pts_B, cam_mat_A, cam_mat_B,
                                                                   dist_cof_A, dist_cof_B)

        np.save("camera_stereo/img_points_stereo_A.npy", img_pts_A)
        np.save("camera_stereo/img_points_stereo_B.npy", img_pts_B)
        np.save("camera_stereo/obj_points_stereo.npy", obj_pts)
        np.save("camera_stereo/Left_Stereo_Map0.npy", Left_Stereo_Map[0])
        np.save("camera_stereo/Left_Stereo_Map1.npy", Left_Stereo_Map[1])
        np.save("camera_stereo/Right_Stereo_Map0.npy", Right_Stereo_Map[0])
        np.save("camera_stereo/Right_Stereo_Map1.npy", Right_Stereo_Map[1])
        np.save("camera_stereo/PL.npy", PL)
        np.save("camera_stereo/PR.npy", PR)

    # Load parameter of stereo pair
    obj_pts = np.load("camera_stereo/obj_points_stereo.npy")
    img_pts_A = np.load("camera_stereo/img_points_stereo_A.npy")
    img_pts_B = np.load("camera_stereo/img_points_stereo_B.npy")
    Left_Stereo_Map = [np.load("camera_stereo/Left_Stereo_Map0.npy"),np.load("camera_stereo/Left_Stereo_Map1.npy")]
    Right_Stereo_Map = [np.load("camera_stereo/Right_Stereo_Map0.npy"),np.load("camera_stereo/Right_Stereo_Map1.npy")]
    PL = np.load("camera_stereo/PL.npy")
    PR = np.load("camera_stereo/PR.npy")

    # Image we want to see
    frameL = openraw("image_khan_g/image_g-11292018094748-0.Raw", 540, 720, 16)
    frameL = np.uint8(frameL * 255)
    frameR = openraw("image_khan_d/image_d-11292018094727-0.Raw", 540, 720, 16)
    frameR = np.uint8(frameR * 255)

    # Rectify the image using the calibration parameters founds during the initialisation
    Left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    Right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    # Resulting rectified image
    plt.imshow(Left_nice, cmap="Greys")
    plt.show()
    plt.imshow(Right_nice, cmap="Greys")
    plt.show()

    # Find centroid of laser spot -> pas encore impolémenté !
    # pos_A = detectpoint(image, threshold)

    # Images coordonate to triangulate
    coor_pts_R = np.array([304, 277], dtype=np.float)
    coor_pts_L = np.array([255, 277], dtype=np.float)

    # Call the triangulation function
    pts_3d = triangulation(PR, PL, coor_pts_R, coor_pts_L)
    print(pts_3d)

    # First iteration for 3d scatter real-time avec update
    for i in range(10):
        y = np.random.random()
        z = np.random.random()
        plt.scatter(i, y, z)
        plt.pause(0.05)
    plt.show()

