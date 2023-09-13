#!/usr/bin/env python3



import cv2
import numpy as np
import math as m


# Function to Convert RGB To Hsi
def RGBtoHSI(image):
    (height, width) = image.shape[:2]
    image = image.astype(float)
    # Seperating RGB Channels and scaling levels from [0...255] to [0...1]
    red = image[:, :, 2]/255
    green = image[:, :, 1]/255
    blue = image[:, :, 0]/255

    hsi = np.zeros((height, width, 3), float) # Declaring initial HSI image

    for i in range(height):
        for j in range(width):

            # Intensity Calculation
            hsi[i][j][2] = (red[i][j]+green[i][j]+blue[i][j])/3

            # Directly Assigning HSI value for Black Color to avoid any division by zero error in future calculations and aviod 0/0 form in Saturation calculation
            if(hsi[i][j][2]==0):
                hsi[i][j][2]=0
                hsi[i][j][1]=0
                hsi[i][j][0]=0
                continue

            # Hue calculation 
            hueNum = float(0.5 * ((red[i][j]-green[i][j])+(red[i][j]-blue[i][j]))) # Numerator
            hueDen = float(pow((pow((red[i][j]-green[i][j]), 2)) + (red[i][j]-blue[i][j])*(green[i][j]-blue[i][j]), 0.5))+0.001 # Denomenator (additional 0.001 to aviod division by zero error)
            hueF = hueNum/hueDen
            hsi[i][j][0] = m.acos(hueF)
            if(blue[i][j] > green[i][j]):
                hsi[i][j][0] = (2*m.pi) - hsi[i][j][0]
            
            # Saturation Calculation
            hsi[i][j][1] = 1 - (min(red[i][j], green[i][j], blue[i][j])/hsi[i][j][2])
    return hsi

### Function for threasholding 

def create_mask(image):
    (height, width) = image.shape[:2]
    masked_image = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            if((0.45< image[i][j][0] and image[i][j][0] < 1) and (0.1 < image[i][j][1] and image[i][j][1] < 0.6) and (0.05 < image[i][j][2] and image[i][j][2] < 0.8)):
                masked_image[i][j] = 255
    # cv2.imshow("IMG_Thresh_",masked_image)
    cv2.imwrite("eleB.jpeg", masked_image)

# For counturing or taking out biggest connected component from the mask


def undesired_objects(image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        image, connectivity=8)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    # cv.imshow("Biggest component", img2)
    # cv.waitKey()
    return img2


# Our Original  Image
real_img = cv2.imread("/home/anshpant/Opencv/temp/Assignment2.jpg", 3)
(height, width) = real_img.shape[:2]



img = RGBtoHSI(real_img)

mask = create_mask(img)
mask = cv2.imread("eleB.jpeg", 0)

# smallMask of Dilation and Erosion
smallMask = np.ones((2, 2), np.uint8)
smallMask2= np.ones((3, 3), np.uint8)
smallMask3 = np.ones((1, 3), np.uint8)



mask = cv2.erode(mask, smallMask, iterations=1)
# cv2.imshow("Erode_ 2x2",mask)
mask = cv2.dilate(mask, smallMask2, iterations=3)
# cv2.imshow("Dilate_3x3",mask)



mask = undesired_objects(mask)
# cv2.imshow("Counturing",mask)

mask = cv2.erode(mask, smallMask, iterations=1)
mask = cv2.dilate(mask, smallMask2, iterations=5)
mask = cv2.erode(mask, smallMask3, iterations=5)

cv2.imshow("Elephant_Mask",mask)
img = cv2.imread("/home/anshpant/Opencv/temp/Assignment2.jpg", 3)

for i in range(height):
    for j in range(width):
        if(mask[i][j] == 0):
            img[i][j][0] = 255
            img[i][j][1] = 255
            img[i][j][2] = 255

cv2.imshow("Final_Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


