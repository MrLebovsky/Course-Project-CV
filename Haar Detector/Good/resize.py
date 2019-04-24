import cv2
import glob, os, errno

i = 0
minW = 222
minH = 353

for fil in glob.glob("*.bmp"):
	image = cv2.imread(fil) 
	height, width, channels = image.shape
	if height < minH:
		minH = height
		
	if width < minW:
		minW = height
		
print('Min Width: ' + str(minW) + ' Min Height: ' + str(minH))
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to greyscale
    #status = cv2.imwrite(r"res/res-%s.jpg"%i,gray_image)
    #print("Image written to file-system : ",status)