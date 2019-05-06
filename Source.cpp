#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;
int main()
{
	Mat seq_1 = imread("0000000001.png");
	Mat seq_2 = imread("0000000000.png");

	Mat descriptor1;
	Mat descriptor2;
	vector<KeyPoint> key_points1, key_points2;
	vector<Mat> descriptor_for_all;
	Mat output_key1, output_key2;

	Ptr<AKAZE> feature = AKAZE::create();

	feature->detect(seq_1, key_points1);
	feature->compute(seq_1, key_points1, descriptor1);
	cout << "Jumlah KeyPoint = " << key_points1.size() << endl;
	drawKeypoints(seq_1, key_points1, output_key1);

	feature->detect(seq_2, key_points2);
	feature->compute(seq_2, key_points2, descriptor2);
	cout << "Jumlah KeyPoint = " << key_points2.size() << endl;
	drawKeypoints(seq_2, key_points2, output_key2);

	if (descriptor1.type() != CV_32F) {
		descriptor1.convertTo(descriptor1, CV_32F);
	}

	if (descriptor2.type() != CV_32F) {
		descriptor2.convertTo(descriptor2, CV_32F);
	}

	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptor1, descriptor2, matches);

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptor1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);


	std::vector< DMatch > good_matches;

	for (int i = 0; i < descriptor1.rows; i++)
	{
		if (matches[i].distance < 300 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	Mat img_matches;
	drawMatches(seq_1, key_points1, seq_2, key_points2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

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

	cout << "H = " << H << endl;

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = Point(0, 0); obj_corners[1] = Point(seq_1.cols, 0);
	obj_corners[2] = Point(seq_1.cols, seq_1.rows); obj_corners[3] = Point(0, seq_1.rows);
	std::vector<Point2f> scene_corners(4);

	perspectiveTransform(obj_corners, scene_corners, H);

	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line(img_matches, scene_corners[0] + Point2f(seq_1.cols, 0), scene_corners[1] + Point2f(seq_1.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[1] + Point2f(seq_1.cols, 0), scene_corners[2] + Point2f(seq_1.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[2] + Point2f(seq_1.cols, 0), scene_corners[3] + Point2f(seq_1.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[3] + Point2f(seq_1.cols, 0), scene_corners[0] + Point2f(seq_1.cols, 0), Scalar(0, 255, 0), 4);

	//-- Show detected matches
	//imshow("Good Matches & Object detection", img_matches);

	cv::Mat result;
	warpPerspective(seq_1, result, H, cv::Size(seq_1.cols + seq_2.cols, seq_1.rows));
	cv::Mat half(result, cv::Rect(0, 0, seq_2.cols, seq_2.rows));
	seq_2.copyTo(half);

	imshow("Result Image", result);
	//imshow("1", output_key1);
	//imshow("2", output_key2);

	waitKey(0);
	return 0;
}
