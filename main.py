from camera_calib import calibrate_camera, rectify_camera, openraw, calibrate_stereo, triangulation, triangulation2, detectpoint, pts_detection
import numpy as np
import os.path
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
        np.save("cameraA/camera_matrix_A", camera_matrix_A)
        np.save("cameraA/distortion_coefficients_A", distortion_coefficients_A)
        np.save("cameraA/rotation_vecs_A", rotation_vecs_A)
        np.save("cameraA/translation_vecs_A", translation_vecs_A)
        np.save("cameraA/img_points_A", img_points_A)
        np.save("cameraA/objects_pts", object_pts)

        print("calibration A done!")
        print(camera_matrix_A,distortion_coefficients_A)

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
        print(camera_matrix_B, distortion_coefficients_B)

    # Load parameter of matrix A -> d
    cam_mat_A = np.load("cameraA/camera_matrix_A.npy")
    dist_cof_A = np.load("cameraA/distortion_coefficients_A.npy")

    # Load parameter of matrix B  -> g
    cam_mat_B = np.load("cameraB/camera_matrix_B.npy")
    dist_cof_B = np.load("cameraB/distortion_coefficients_B.npy")

    if (os.path.isfile("camera_stereo/img_points_stereo_B.npy") == False):

        os.mkdir('camera_stereo')

        obj_pts, img_pts_A, img_pts_B = calibrate_stereo(15, 10, "calib_stereo_d", "calib_stereo_g")
        (Left_Stereo_Map, Right_Stereo_Map, PL, PR) = rectify_camera(obj_pts, img_pts_A, img_pts_B, cam_mat_A, cam_mat_B, dist_cof_A, dist_cof_B)

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
    Left_Stereo_Map = [np.load("camera_stereo/Left_Stereo_Map0.npy"), np.load("camera_stereo/Left_Stereo_Map1.npy")]
    Right_Stereo_Map = [np.load("camera_stereo/Right_Stereo_Map0.npy"), np.load("camera_stereo/Right_Stereo_Map1.npy")]
    PL = np.load("camera_stereo/PL.npy")
    PR = np.load("camera_stereo/PR.npy")

    # Image we want to see
    frameL = openraw("D:\Desktop\Projet - v2\image_stereo_g\image-g-12162018141044-0.Raw", 540, 720, 16)
    frameL = np.uint8(frameL * 255)
    frameR = openraw("D:\Desktop\Projet - v2\image_stereo_d\stereo-d-12162018141044-0.Raw", 540, 720, 16)
    frameR = np.uint8(frameR * 255)

    # Rectify the image using the calibration parameters founds during the initialisation
    Left_nice = cv2.equalizeHist(cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0))
    Right_nice = cv2.equalizeHist(cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0))

    # Resulting rectified image
    plt.imshow(Left_nice, cmap="Greys")
    plt.show()
    plt.imshow(Right_nice, cmap="Greys")
    plt.show()

    triang_image = False
    if triang_image == True:
        # Images coordonate to triangulate
        pts_list_d, pts_list_g = pts_detection("D:\Desktop\Projet - v2\image_tasse_d", "D:\Desktop\Projet - v2\image_tasse_g")
        #pts_list_d, pts_list_g = pts_detection("D:\Desktop\Projet - v2\image_d", "D:\Desktop\Projet - v2\image_g")
        pts_3d_list = []

        for (pts_d, pts_g) in zip(pts_list_d, pts_list_g):
            if np.abs(pts_d[1]-pts_g[1]) < 3 and pts_d[1] != 0 and pts_g[1] != 0:
                pts3d = triangulation(PR, PL, pts_d, pts_g)
                pts_3d_list.append(pts3d)
            else:
                pass


        # First iteration for 3d scatter real-time avec update
        x, y, z = [], [], []

        for pts in pts_3d_list:
            x.append(pts[0])
            y.append(-pts[1])
            z.append(pts[2])

        np.save("pts_3d/x.npy", x)
        np.save("pts_3d/y.npy", y)
        np.save("pts_3d/z.npy", z)

    perf_calib = True
    if perf_calib == True:
        #points obtenus dans FIJI sur des images _g _d correspondantes d'un damier.
        #horizontal
        # list_g = np.array([[118,348],[146,348],[173,348],[201,348],[229,349],[258,349],[287,349],[316,349],[346,349],[376,349]], dtype=np.float)#np.array([[185, 168], [212, 168], [238, 167], [264, 165], [292, 164], [319, 162], [347, 162], [374, 162], [402, 159],[430, 159], [457, 158], [486, 157], [513, 156]], dtype=np.float)
        # list_d = np.array([[128,348],[149,348],[170,348],[191,348],[213,348],[235,348],[257,348],[280,349],[302,349],[325,349]], dtype=np.float)#np.array([[146,  156], [165, 155], [183, 154], [203, 153], [224, 151], [244, 149], [265, 147], [286, 145], [307, 144],[330, 143], [351, 141], [375, 139], [399, 136]], dtype=np.float)

        #vertical
        list_g = np.array([[533,71],[533,102],[533,133],[534,164],[535,195],[534,226],[534,256],[535,287],[535,317],[535,347]],dtype=np.float)
        list_d = np.array([[441,72],[442,103],[442,134],[443,164],[443,195],[443,225],[444,256],[445,287],[445,317],[446,348],], dtype=np.float)

        pts_3d_list2 = []
        for (pts_d, pts_g) in zip(list_d, list_g):
            pts3d = triangulation2(PR, PL, pts_d, pts_g)
            pts_3d_list2.append(pts3d)

        x1, y1, z1 = [], [], []
        for pts in pts_3d_list2:
            x1.append(pts[0]/pts[3])
            y1.append(-pts[1]/pts[3])
            z1.append(pts[2]/pts[3])

        np.save("pts_3d/x1.npy", x1)
        np.save("pts_3d/y1.npy", y1)
        np.save("pts_3d/z1.npy", z1)

    graph = True
    if graph == True:
        ax.scatter(x, y, z, s=10, depthshade=True)
        plt.ylim(5, 6)
        plt.show()

        ax.scatter(x, y, z, s=10, depthshade=True)
        plt.show()

        im = plt.scatter(x, y, s=15, c=z, marker='o')#, cmap=plt.cm.get_cmap('RdYlBu'))
        plt.xlabel("Position x [cm]", fontsize=14)
        plt.ylabel("Position y [cm]", fontsize=14)
        plt.colorbar(im, label="Position en z [cm]")#, fontsize=14)
        plt.show()

        im1 = plt.scatter(x, z, s=15, c=y, marker='o')#, cmap=plt.cm.get_cmap('RdYlBu'))
        plt.xlabel("Position x [cm]", fontsize=14)
        plt.ylabel("Position z [cm]", fontsize=14)
        plt.colorbar(im1, label="Position en y [cm]")
        #plt.colorbar.set_label(label="Position en y [cm]")#, size=14)
        plt.show()

        im2 = plt.scatter(y, z, s=15, c=x, marker='o', cmap=plt.cm.get_cmap('RdYlBu'))
        plt.colorbar(im2)
        plt.show()