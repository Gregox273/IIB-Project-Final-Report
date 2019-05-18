import cv2
import numpy as np

# Gregory Brooks 2019
# Based on tutorial by Kristijan Ivancic at https://realpython.com/traditional-face-detection-python/ (accessed 18 May 2019)

def outline_faces(image, face_cascade):
	faces = face_cascade.detectMultiScale(image)
	for(column, row, width, height) in faces:
		cv2.rectangle(
			image,
			(column, row),
			(column + width, row + height),
			(0, 255, 0),
			2
		)
	println(faces)
	return image
	
def println(arg):
	print(arg, "\n")


# Read original image as greyscale
original_image = cv2.imread('face.jpg', 0)

# Resize
resized_image = cv2.resize(original_image, (32,32), interpolation = cv2.INTER_NEAREST)
#cv2.imwrite('resized.jpg', resized_image)

# Random noise added to pixel intensities
image = resized_image
row,col= image.shape
noise = np.random.laplace(0,25,row*col)
noise = noise.reshape(row,col)
noise = noise.astype(int)
noised_image = np.uint8(np.clip(image + noise, 0, 255))
#cv2.imwrite('noised_image.jpg', noised_image)

println(noise)
println(image)
println(noised_image)

# Viola-Jones
resized_image = cv2.resize(resized_image, (320, 320), interpolation = cv2.INTER_NEAREST)  # Allow rectangle to be drawn on
noised_image = cv2.resize(noised_image, (320, 320), interpolation = cv2.INTER_NEAREST)    # Allow rectangle to be drawn on

face_cascade = cv2.CascadeClassifier('/usr/local/lib/python3.5/dist-packages/cv2/data/haarcascade_frontalface_alt.xml')

resized_image = outline_faces(resized_image, face_cascade)
noised_image = outline_faces(noised_image, face_cascade)

cv2.imwrite('resized_image.jpg', resized_image)
cv2.imwrite('noised_image.jpg', noised_image)

#cv2.imshow('test',resized_image)
cv2.imshow('test',noised_image)

cv2.waitKey(0)
cv2.destroyAllWindows()



