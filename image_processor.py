import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('sample.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('grayscale_sample.jpg', gray_image)

# Resize the image
resized_image = cv2.resize(image, (100, 100))
cv2.imwrite('resized_sample.jpg', resized_image)

# Apply Gaussian blur
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
cv2.imwrite('blurred_sample.jpg', blurred_image)

# Combine the processed images into a single image for saving
# Convert grayscale to BGR to match the color format
gray_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
# Resize images to have the same height
resized_gray = cv2.resize(gray_bgr, (image.shape[1], image.shape[0]))
resized_resized = cv2.resize(resized_image, (image.shape[1], image.shape[0]))
resized_blurred = cv2.resize(blurred_image, (image.shape[1], image.shape[0]))

# Concatenate images horizontally
final_image = np.hstack((resized_gray, resized_resized, resized_blurred))
cv2.imwrite('final_image.jpg', final_image)

# Display the images
plt.figure(figsize=(12, 6))

# Display grayscale image
plt.subplot(1, 3, 1)
plt.title("Grayscale Image")
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

# Display resized image
plt.subplot(1, 3, 2)
plt.title("Resized Image")
plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Display blurred image
plt.subplot(1, 3, 3)
plt.title("Blurred Image")
plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Show the images
plt.tight_layout()
plt.show()

# Display and save the final image
plt.figure(figsize=(12, 6))
plt.title("Final Combined Image")
plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
