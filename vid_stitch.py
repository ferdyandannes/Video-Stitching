import argparse
import logging

import cv2

import image_stitching
from image_stitching import combine
from image_stitching import helpers
from image_stitching import matching


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video_path', type=str, nargs='+', help="paths to one or more images or image directories")
    parser.add_argument('-b', '--debug', dest='debug', action='store_true', help='enable debug logging')
    parser.add_argument('-q', '--quiet', dest='quiet', action='store_true', help='disable all logging')
    parser.add_argument('-d', '--display', dest='display', action='store_true', help="display result")
    parser.add_argument('-s', '--save', dest='save', action='store_true', help="save result to file")
    parser.add_argument("--save_path", dest='save_path', default="stitched.png", type=str, help="path to save result")
    parser.add_argument('-k', '--knn', dest='knn', default=2, type=int, help="Knn cluster value")
    parser.add_argument('-l', '--lowe', dest='lowe', default=0.7, type=float, help='acceptable distance between points')
    parser.add_argument('-m', '--min', dest='min_correspondence', default=10, type=int, help='min correspondences')
    args = parser.parse_args()

    result = None
    result_gry = None

    #print('vid = ', video_path)

    flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})

    #cap = cv2.VideoCapture(args.video_path)
    cap = cv2.VideoCapture('kitti.mp4')

    while(cap.isOpened()):
        ret, currImage = cap.read()
        currImage_gray = cv2.cvtColor(currImage, cv2.COLOR_BGR2GRAY)

        if result is None :
            print('a')
            result = currImage
            img4 = result
            img5 = result
            img6 = result
        else :
            print('ada1')
            sift = cv2.xfeatures2d.SIFT_create()
            features0 = sift.detectAndCompute(result, None)
            features1 = sift.detectAndCompute(currImage_gray, None)

            keypoints0, descriptors0 = features0
            keypoints1, descriptors1 = features1

            img4 = cv2.drawKeypoints(result, keypoints0, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            img5 = cv2.drawKeypoints(currImage, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            matches_src, matches_dst, n_matches, matchezz, matchesMask = image_stitching.compute_matches(
                features0, features1, flann, knn=args.knn)

            draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0),matchesMask = matchesMask, flags = 0)
            img6 = cv2.drawMatchesKnn(result, keypoints0, currImage, keypoints1, matchezz, None, **draw_params)

            H, mask = cv2.findHomography(matches_src, matches_dst, cv2.RANSAC, 5.0)
            result = image_stitching.combine_images(currImage, result, H)

            #cv2.imshow('fast_true1.png', img4)
            #cv2.waitKey(0)

        print('show')
        #cv2.imshow('fast_true1.png', img4)
        #cv2.imshow('fast_true2.png', img5)
        #cv2.imshow('curr',currImage)
        #cv2.imshow('prev',result)
        cv2.imwrite('match.png', img6)
        cv2.imwrite('result.png', result)
        cv2.waitKey(1)

    
    cap.release()
    cv2.waitKey(0)
