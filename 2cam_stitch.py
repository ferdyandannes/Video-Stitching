import argparse
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})
sift = cv2.xfeatures2d.SIFT_create()

cap_1 = cv2.VideoCapture('seq2.mp4')
cap_2 = cv2.VideoCapture('seq1.mp4')

def trim(frame):
    if not np.sum(frame[0]):
        return trim(frame[1:])
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

while(cap_1.isOpened()) :
    ret_1, seq_1 = cap_1.read()
    seq_1_gray = cv2.cvtColor(seq_1, cv2.COLOR_BGR2GRAY)
    
    ret_2, seq_2 = cap_2.read()
    seq_2_gray = cv2.cvtColor(seq_2, cv2.COLOR_BGR2GRAY)

    features0 = sift.detectAndCompute(seq_1_gray, None)
    features1 = sift.detectAndCompute(seq_2_gray, None)

    keypoints0, descriptors0 = features0
    keypoints1, descriptors1 = features1

    #img4 = cv2.drawKeypoints(seq_1, keypoints0, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #img5 = cv2.drawKeypoints(seq_2, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    matches = flann.knnMatch(descriptors0, descriptors1, k=2)

    good = []
    for m in matches:
        if m[0].distance < 0.7*m[1].distance:         
            good.append(m)
    matches = np.asarray(good)
    
    #draw_params = dict(matchColor = (0, 255, 0), singlePointColor = (255, 0 ,0), matchesMask = matchesMask, flags = 0)
    #img6 = cv2.drawMatchesKnn(seq_1, keypoints0, seq_2, keypoints1, matchezz, None, **draw_params)

    if len(matches[:,0]) >= 4:
        src = np.float32([ keypoints0[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ keypoints1[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    else:
        raise AssertionError("Can’t find enough keypoints.")

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    dst = cv2.warpPerspective(seq_1, H, (seq_2.shape[1] + seq_1.shape[1], seq_2.shape[0]))

    cv2.imshow("original_image_stitched_crop.jpg", trim(dst))
    cv2.waitKey(0)
        
cap_1.release()
cap_2.release()
cv2.waitKey(0)
