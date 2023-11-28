import cv2 as cv
import numpy as np
import os

# change working directory to the folder this script is in
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# load images
haystack_img = cv.imread('mine.png', cv.IMREAD_UNCHANGED)
needle_img = cv.imread('copper.png', cv.IMREAD_UNCHANGED)

result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)
print(result)

threshold = 0.39
locations = np.where(result >= threshold)
# zip will create tuples of x, y coordinates
locations = list(zip(*locations[::-1]))
# print(locations)

if locations:
  print('Found needle.')

  needle_w = needle_img.shape[1]
  needle_h = needle_img.shape[0]
  line_color = (0, 255, 0)
  line_type = cv.LINE_4

  # loop over all the locations and draw their rectangles
  for loc in locations:
    # determine the box positions
    top_left = loc
    bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)
    # draw the box
    cv.rectangle(haystack_img, top_left, bottom_right, line_color, line_type)

  cv.imshow('Matches', haystack_img)
  cv.waitKey()
  cv.imwrite('result.png', haystack_img)