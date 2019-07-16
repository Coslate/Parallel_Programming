#ifndef LIMBUS_FEATUREDETECTION_H
#define LIMBUS_FEATUREDETECTION_H
#pragma once

#include <opencv2\core\core.hpp>  
#include <opencv2\opencv.hpp>
#include<iostream>
#include <opencv2\highgui.hpp>
//#include <highgui.h>
//#include <cv.h>
#include "Parameters.h"

using namespace std;
using namespace cv;


bool checkpoint(int x , int y);

void LimbusFeatureDetection(const Mat &in , vector<Point> &feature, const int num_per_line 
	, Point & start_point , const bool setup_eye_start_point_man , const Mat &Iris_Mask);

void LimbusFeatureDetection(const Mat &in, vector<Point> &feature, const int num_per_line
	, Point & start_point, const bool setup_eye_start_point_man, const Mat &Iris_Mask, vector<double> &time_eye_position_detection_limbus_feature_detection_serial);

void ParalleLimbusFeatureDetection(const Mat &in, vector<Point> &feature, const int num_per_line
	, Point & start_point, const bool setup_eye_start_point_man, const Mat &Iris_Mask, vector<double> &time_eye_position_detection_limbus_feature_detection_serial, const int thread_num);
#endif