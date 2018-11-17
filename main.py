from camera_calib import calibrate_camera
import cv2
import numpy as np
import os.path

if __name__ == '__main__':
    

    if (os.path.isfile("camera_matrix_A.npy") == False):
        (ret_A,
        camera_matrix_A,
        distortion_coefficients_A,
        rotation_vecs_A,
        translation_vecs_A) = calibrate_camera(6, 9, './calibA/*.jpg')

        np.save("camera_matrix_A",camera_matrix_A)
        np.save("rotation_vecs_A", rotation_vecs_A)
        np.save("translation_vecs_B", translation_vecs_A)
        print("calibration A done!")


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

