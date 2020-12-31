import cv2

img = cv2.imread('test6.jpg')
dst_image_name = "test_scratch3_3.jpg"
dst_gray, dst_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)

cv2.imwrite(dst_image_name, dst_color)