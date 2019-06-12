#ifndef BLINK_DETECTION_H
#define BLINK_DETECTION_H
#pragma once


#include <opencv2\core\core.hpp>  
#include <opencv2\opencv.hpp>
#include <iostream>
#include <opencv2\highgui.hpp>
//#include <highgui.h>
#include <vector>
//#include <cv.h>
#include "Parameters.h"

using namespace std;
using namespace cv;

enum {
	Eyeclose = false,
	Eyeopen = true
};


void EyeBlinkDetection(const bool &noLimbusFeaturePts  , const bool &caculateIris_Mask_done								
								, bool &eyeState , bool &pre_eyeState 
								, int &voluntaryBlinkCount , int &non_voluntaryBlinkCount 
								, int &countCloseFrame
								, const vector<Point> &IrisContoursPoints , const bool &getIrisContourPoints
								, const bool &extremeRight_forBlink, const bool &extremeDown_forBlink 
								, const bool &extremeUp_forBlink , const bool &extremeLeft_forBlink
								, deque<float> &ratio_queue 
								, const int &irisContour_size
								, const float &eyeCloseDetermine_irisContourSizeThreshold_colorModelBased);


#endif