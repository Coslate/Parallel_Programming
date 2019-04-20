#ifndef LIMBUS_FEATUREDETECTION_H
#define LIMBUS_FEATUREDETECTION_H
#pragma once

#include <opencv2\core\core.hpp>  
#include <opencv2\opencv.hpp>
#include<iostream>
//#include <highgui.h>
//#include <cv.h>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui.hpp>
#include "Parameters.h"

using namespace std;
using namespace cv;


bool checkpoint(int x , int y);

void LimbusFeatureDetection(const Mat &in , vector<Point> &feature , const int frame_number , const int num_per_line 
	, Point & start_point , const bool setup_eye_start_point_man , const Mat &Iris_Mask);


#endif