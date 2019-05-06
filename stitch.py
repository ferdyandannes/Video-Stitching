import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

def combine_images(img0, img1, h_matrix):
    points0 = np.array(
        [[0, 0], [0, img0.shape[0]], [img0.shape[1], img0.shape[0]], [img0.shape[1], 0]], dtype=np.float32)
    points0 = points0.reshape((-1, 1, 2))
    points1 = np.array(
        [[0, 0], [0, img1.shape[0]], [img1.shape[1], img0.shape[0]], [img1.shape[1], 0]], dtype=np.float32)
    points1 = points1.reshape((-1, 1, 2))
    points2 = cv2.perspectiveTransform(points1, h_matrix)
    points = np.concatenate((points0, points2), axis=0)
    [x_min, y_min] = np.int32(points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(points.max(axis=0).ravel() + 0.5)
    H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    output_img = cv2.warpPerspective(img1, H_translation.dot(h_matrix), (x_max - x_min, y_max - y_min))
    output_img[-y_min:img0.shape[0] - y_min, -x_min:img0.shape[1] - x_min] = img0
    return output_img

img_ = cv2.imread('2.jpeg')
img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)

img = cv2.imread('1.jpeg')
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN Matcher
flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})
matches = flann.knnMatch(des1,des2,k=2)

# Apply ratio test
good = []
for m in matches:
     if m[0].distance < 0.7*m[1].distance:         
     	good.append(m)
matches = np.asarray(good)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2, outImg = None)
 	 
good = []
for m in matches:
     if m[0].distance < 0.7*m[1].distance:
          good.append(m)
matches = np.asarray(good)

if len(matches[:,0]) >= 4:
     src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
     dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
     H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
else:
     raise AssertionError("Can’t find enough keypoints.")

H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
   
dst = cv2.warpPerspective(img_, H, (img.shape[1] + img_.shape[1], img.shape[0]))
result = combine_images(img, img_, H)

plt.subplot(122),plt.imshow(dst),plt.title('Warped Image')
plt.show()
plt.figure()
dst[0:img.shape[0], 0:img.shape[1]] = img
plt.imshow(dst)
plt.show()

cv2.imshow('resultant_stitched_panorama.jpg', dst)
cv2.waitKey(0)

def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame
cv2.imshow("original_image_stitched_crop.jpg", trim(dst))
cv2.waitKey(0)
