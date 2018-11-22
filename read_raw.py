import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import StringIO

#img = cv2.imread("2_CCD1.raw", 0)


def openraw(namefile, width, heigth, bit):

	#Ouverture des fichiers d'image
	# with open(namefile,'r', encoding='utf-8', errors='ignore') as fdata:

	# 	signal = np.fromfile(fdata, np.uint16).reshape((width,heigth))

	# with open(namebackground,'r', encoding='utf-8', errors='ignore') as fdata:
	
	# 	background = np.fromfile(fdata, np.uint16).reshape((width,heigth))

	# Soustraction de l'image de fond à l'image du signal 
	# background = background * np.ones([width, heigth])

	image = np.fromfile(namefile, dtype=np.uint16)
	image.shape = (width, heigth)
	outputImg8U = cv2.convertScaleAbs(image, alpha=(255.0/65535.0))
	# signal = signal * np.ones([width, heigth])/(2**bit)
	
	return outputImg8U


# img = openraw("/Users/antoine/Documents/Vision num/VisionNum3DRecons/Calib_d/calib_d-11212018152545-0.Raw", 540, 720, 16)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




