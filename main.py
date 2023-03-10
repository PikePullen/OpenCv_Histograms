import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_image():
    blank_img = np.zeros((600,600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank_img, text='ABCDE', org=(50,300), fontFace=font, fontScale=5, color=(255,255,255), thickness=25)
    return blank_img

def display_img(img, cmap=None):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap=cmap)
    plt.show()

"""Sample images"""
# dark_horse = cv2.imread('../DATA/horse.jpg') #ORIGINAL BGR OPENCV
# show_horse = cv2.cvtColor(dark_horse, cv2.COLOR_BGR2RGB) #CONVERTED TO RGB FOR SHOW

# rainbow = cv2.imread('../DATA/rainbow.jpg') #ORIGINAL BGR OPENCV
# show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB) #CONVERTED TO RGB FOR SHOW

# blue_bricks = cv2.imread('../DATA/bricks.jpg') #ORIGINAL BGR OPENCV
# show_bricks = cv2.cvtColor(blue_bricks, cv2.COLOR_BGR2RGB) #CONVERTED TO RGB FOR SHOW

# display_img(show_horse)
# display_img(show_rainbow)
# display_img(show_bricks)

"""display the blue channel values for bricks"""
# hist_values = cv2.calcHist([blue_bricks], channels=[0], mask=None, histSize=[256], ranges=[0,256])
# print(hist_values.shape)
# plt.plot(hist_values)
# plt.show()

"""display the blue channel values for horse"""
# hist_values2 = cv2.calcHist([dark_horse], channels=[0], mask=None, histSize=[256], ranges=[0,256])
# print(hist_values2.shape)
# plt.plot(hist_values2)
# plt.show()

"""display all the histogram channels"""
# img = blue_bricks
# color = ('b', 'g', 'r')
#
# for i,col in enumerate(color):
#     histr = cv2.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr, color=col)
#     plt.xlim([0,256]) # set the scale and zoom on the histogram graph
#
# plt.title('HISTOGRAM FOR BLUE BRICKS')
# plt.show()

"""display all the histogram channels"""
# img = dark_horse
# color = ('b', 'g', 'r')
#
# for i,col in enumerate(color):
#     histr = cv2.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr, color=col)
#     plt.xlim([0,50]) # set the scale and zoom on the histogram graph
#     plt.ylim([0,500000]) # set the scale and zoom on the histogram graph
#
# plt.title('HISTOGRAM FOR HORSE')
# plt.show()

"""
Histogram Equalization is a method of contrast adjustment based on the image's histogram
going from low contrast becomes high contrast with equalization
"""

# rainbow = cv2.imread('../DATA/rainbow.jpg') #ORIGINAL BGR OPENCV
# show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB) #CONVERTED TO RGB FOR SHOW
# print(show_rainbow.shape)
# display_img(show_rainbow)

# img = rainbow

# create black image
# mask = np.zeros(img.shape[:2], np.uint8)
# plt.imshow(mask,cmap='gray')
# plt.show()

# create white section of black image
# mask[300:400,100:400] = 255
# plt.imshow(mask,cmap='gray')
# plt.show()

# apply mask to rainbow
# masked_img = cv2.bitwise_and(img, img, mask=mask)
# show_masked_img = cv2.bitwise_and(show_rainbow, show_rainbow, mask=mask)
# plt.imshow(show_masked_img)
# plt.show()

# hist_mask_values_red = cv2.calcHist([rainbow], channels=[2], mask=mask, histSize=[256], ranges=[0,256])
# hist_values_red = cv2.calcHist([rainbow], channels=[2], mask=None, histSize=[256], ranges=[0,256])

# plt.plot(hist_mask_values_red)
# plt.title('RED HISTOGRAM FOR MASKED RAINBOW')
# plt.show()

# plt.plot(hist_values_red)
# plt.title('RED HISTOGRAM NO MASK, FOR RAINBOW')
# plt.show()

"""Last Histogram Session"""
# gorilla = cv2.imread('../DATA/gorilla.jpg',0) #ORIGINAL BGR OPENCV
# print(gorilla.shape)
# display_img(gorilla, cmap='gray')
#
# # histogram of original grayscale image
# hist_values = cv2.calcHist([gorilla], channels=[0], mask=None, histSize=[256], ranges=[0,256])
# plt.plot(hist_values)
# plt.show()
#
# # histogram of equalized version of the grayscale image, increases contrast a lot
# eq_gorilla = cv2.equalizeHist(gorilla)
# display_img(eq_gorilla, cmap='gray')
#
# hist_values2 = cv2.calcHist([eq_gorilla], channels=[0], mask=None, histSize=[256], ranges=[0,256])
# plt.plot(hist_values2)
# plt.show()

"""equalize a color image"""
# color_gorilla = cv2.imread('../DATA/gorilla.jpg')
# show_gorilla = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2RGB)
# display_img(show_gorilla)
#
# hsv = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2HSV)
# hsv[:,:,2].min()
# hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
#
# eq_color_gorilla = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
# display_img(eq_color_gorilla)