from camera_calib import calibrate_camera

if __name__ == '__main__':
    
    (ret,
    camera_matrix,
    distortion_coefficients,
    rotation_vecs,
    translation_vecs) = calibrate_camera(6, 9)
