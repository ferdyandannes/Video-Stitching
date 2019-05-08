// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <opencv2/ximgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <stdio.h>
#include <chrono>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main()
{
	VideoCapture cap_1("seq2.mp4");
	VideoCapture cap_2("seq1.mp4");

	if (!cap_1.isOpened()) {
		cout << "No video 1 found" << endl;
		return -1;
	}

	if (!cap_2.isOpened()) {
		cout << "No video 2 found" << endl;
		return -1;
	}

	while (1) {
		Mat seq_1, seq_2;
		auto start = std::chrono::system_clock::now();
		cap_1 >> seq_1;
		cap_2 >> seq_2;

		if (seq_1.empty())
		{
			break;
		}

		if (seq_2.empty())
		{
			break;
		}

		Mat descriptor1;
		Mat descriptor2;
		vector<KeyPoint> key_points1, key_points2;
		vector<Mat> descriptor_for_all;
		Mat output_key1, output_key2;

		Ptr<AKAZE> feature = AKAZE::create();

		int minHessian = 400;
		Ptr<SURF> detector = SURF::create(minHessian);


		detector->detect(seq_1, key_points1);
		detector->compute(seq_1, key_points1, descriptor1);
		//cout << "Jumlah KeyPoint 1 = " << key_points1.size() << endl;
		//drawKeypoints(seq_1, key_points1, output_key1);

		detector->detect(seq_2, key_points2);
		detector->compute(seq_2, key_points2, descriptor2);
		//cout << "Jumlah KeyPoint 2 = " << key_points2.size() << endl;
		//drawKeypoints(seq_2, key_points2, output_key2);

		if (descriptor1.type() != CV_32F) {
			descriptor1.convertTo(descriptor1, CV_32F);
		}

		if (descriptor2.type() != CV_32F) {
			descriptor2.convertTo(descriptor2, CV_32F);
		}

		FlannBasedMatcher matcher;
		std::vector< DMatch > matches;
		matcher.match(descriptor1, descriptor2, matches);

		double max_dist = 0; double min_dist = 2000;

		//-- Quick calculation of max and min distances between keypoints
		for (int i = 0; i < descriptor1.rows; i++)
		{
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		//printf("-- Max dist : %f \n", max_dist);
		//printf("-- Min dist : %f \n", min_dist);

		std::vector< DMatch > good_matches;

		for (int i = 0; i < descriptor1.rows; i++)
		{
			if (matches[i].distance < 50 * min_dist)
			{
				good_matches.push_back(matches[i]);
			}
		}

		//Mat img_matches;
		//drawMatches(seq_1, key_points1, seq_2, key_points2,
			//good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
			//vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		//-- Localize the object
		std::vector<Point2f> obj;
		std::vector<Point2f> scene;

		for (int i = 0; i < good_matches.size(); i++)
		{
			//-- Get the keypoints from the good matches
			obj.push_back(key_points1[good_matches[i].queryIdx].pt);
			scene.push_back(key_points2[good_matches[i].trainIdx].pt);
		}

		Mat H = findHomography(obj, scene, RANSAC);

		//cout << "H = " << H << endl;

		cv::Mat result;
		warpPerspective(seq_1, result, H, cv::Size(seq_1.cols + seq_2.cols, seq_1.rows));
		cv::Mat half(result, cv::Rect(0, 0, seq_2.cols, seq_2.rows));
		seq_2.copyTo(half);

		imshow("Result Image", result);
		//imshow("Seq 1", output_key1);
		//imshow("Seq 2", output_key2);

		//imwrite("ResultImage.jpg", result);
		//imwrite("Seq1.jpg", output_key1);
		//imwrite("Seq2.jpg", output_key2);

		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> diff = end - start;
		std::cout << "Time to process last frame (seconds): " << diff.count()
			<< " FPS: " << 1.0 / diff.count() << endl;

		if ((char)cv::waitKey(33) >= 0) break;
	}
	cap_1.release();
	cap_2.release();
	destroyAllWindows();
	waitKey(1);
	return 0;
}