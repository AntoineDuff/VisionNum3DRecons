from camera_calib import calibrate_camera
import cv2

if __name__ == '__main__':
    
    # (ret,
    # camera_matrix,
    # distortion_coefficients,
    # rotation_vecs,
    # translation_vecs) = calibrate_camera(6, 9)

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

