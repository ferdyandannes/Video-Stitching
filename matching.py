#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Will Brennan'

import logging

import cv2
import numpy as np

logger = logging.getLogger("main")


def compute_matches(features0, features1, matcher, knn=2, lowe=0.7):
    keypoints0, descriptors0 = features0
    keypoints1, descriptors1 = features1

    logger.debug('finding correspondence')

    matches = matcher.knnMatch(descriptors0, descriptors1, k=knn)

    #matchesMask = [[0,0] for i in range(len(matches))]
    
    logger.debug("filtering matches with lowe test")

    positive = []
    for match0, match1 in matches:
        if match0.distance < lowe * match1.distance:
            positive.append(match0)

    # tambahan #1
    matchez = np.asarray(positive)

    # tambahan #2
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matchezz = flann.knnMatch(descriptors0,descriptors1,k=2)
    matchesMask = [[0,0] for i in range(len(matchezz))]

    for i,(m,n) in enumerate(matchezz):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]




    src_pts = np.array([keypoints0[good_match.queryIdx].pt for good_match in positive], dtype=np.float32)
    src_pts = src_pts.reshape((-1, 1, 2))
    dst_pts = np.array([keypoints1[good_match.trainIdx].pt for good_match in positive], dtype=np.float32)
    dst_pts = dst_pts.reshape((-1, 1, 2))

    return src_pts, dst_pts, len(positive), matchezz, matchesMask
