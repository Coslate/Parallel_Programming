#ifndef EYELID_FEATUREDETECTION_H
#define EYELID_FEATUREDETECTION_H
#pragma once

#include <opencv2\core\core.hpp>  
#include <opencv2\opencv.hpp>
#include <iostream>
#include <opencv2\highgui.hpp>
//#include <highgui.h>
//#include <cv.h>
#include "Parameters.h"

using namespace std;
using namespace cv;


void EyelidFeatureDetection(const Mat &Src , vector<Point> &upperEyelid_feature , vector<Point> &lowerEyelid_feature 
	, const int &frame_number	, const int &eyeRegionCenter_y , const Mat &ABS_Grad_X_mask);


#endif