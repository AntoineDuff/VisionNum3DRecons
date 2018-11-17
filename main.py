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
        translation_vecs_A) = calibrate_camera(6, 9, './calibA/*.jpg')

        np.save("cameraA/ret_A", ret_A)
        np.save("cameraA/camera_matrix_A",camera_matrix_A)
        np.save("cameraA/distortion_coefficients_A", distortion_coefficients_A)
        np.save("cameraA/rotation_vecs_A", rotation_vecs_A)
        np.save("cameraA/translation_vecs_A", translation_vecs_A)
        print("calibration A done!")

        if (os.path.isfile("cameraB/camera_matrix_B.npy") == False):

            os.mkdir('cameraB')
            (ret_B,
            camera_matrix_B,
            distortion_coefficients_B,
            rotation_vecs_B,
            translation_vecs_B) = calibrate_camera(6, 9, './calibB/*.jpg')

            np.save("cameraB/cret_B", ret_A)
            np.save("cameraB/ccamera_matrix_B",camera_matrix_A)
            np.save("cameraB/cdistortion_coefficients_B", distortion_coefficients_A)
            np.save("cameraB/crotation_vecs_B", rotation_vecs_A)
            np.save("cameraB/ctranslation_vecs_B", translation_vecs_A)
            print("calibration B done!")

    # tt = np.load("rotation_vecs_A.npy")
    # yy = np.load("camera_matrix_A.npy")
    # print("ok")

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

