'''
	The script is for converting sequential images to a single video (.avi) file.

	Replace directory to the one containing images
	Output is converted_video.avi in current directory

	You can specify sleep and next image selection time
'''

import cv2
import numpy as np 
import glob
import os
import time 
from datetime import datetime

images = []

# make sure there are only images in the directory, code is not written to filter files
abs_dir = '/path/to/images/'

files = os.listdir(abs_dir)
files.sort()

for file in files:
	img = cv2.imread(abs_dir + file)
	images.append(img)


height, width, channels = images[0].shape
size = (width, height)

out = cv2.VideoWriter('converted_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

for i in range(len(images)):
	start = datetime.now()
	while True:
		out.write(images[i])
		delta = datetime.now() - start
		if delta.total_seconds() >= 0.135:
			break
		else:
			time.sleep(0.02)

out.release()