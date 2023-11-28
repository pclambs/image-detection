import cv2 as cv
import numpy as np

# load images
haystack_img = cv.imread('mine.png', cv.IMREAD_UNCHANGED)
needle_img = cv.imread('copper.png', cv.IMREAD_UNCHANGED)

result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)

# get the best match position
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

print('Best match top left position: %s' % str(max_loc))
print('Best match confidence: %s' % max_val)

# threshold to decide when we have found a match
threshold = 0.8
if max_val >= threshold:
  print('found needle')

  # get dimensions of the needle image
  needle_w = needle_img.shape[1]
  needle_h = needle_img.shape[0]

  # get dimensions of the needle image
  top_left = max_loc
  bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)

  # draw a rectangle around the matched region
  cv.rectangle(haystack_img, top_left, bottom_right, 
                color=(0, 255, 0), thickness=2, lineType=cv.LINE_4)
  
  # show the result
  # cv.imshow('Result', haystack_img)

  # save the result
  cv.imwrite('result.png', haystack_img)

  cv.waitKey()
else:
  print('no needle found')