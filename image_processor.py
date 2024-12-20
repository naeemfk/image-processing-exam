import cv2

# Read the image
image = cv2.imread('sample.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Save the grayscale image
cv2.imwrite('grayscale_sample.jpg', gray_image)
