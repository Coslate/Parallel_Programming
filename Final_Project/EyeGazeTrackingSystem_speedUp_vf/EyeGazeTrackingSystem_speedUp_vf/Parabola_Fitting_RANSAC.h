#ifndef PARABOLA_FITTING_RANSAC_H
#define PARABOLA_FITTING_RANSAC_H
#pragma once
#define _USE_MATH_DEFINES 

#include <opencv2\core\core.hpp>  
#include <opencv2\opencv.hpp>
#include <iostream>
//#include <highgui.h>
//#include <cv.h>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui.hpp>
#include <cstdio>
#include <omp.h>
#include <fstream>
#include "Parameters.h"
#include "SVD.h"

using namespace cv;
using namespace std;


void Parabola_Fitting_RANSACUp(const Mat &Src , const vector<Point> &feature_point ,float* &parabola_param
											, vector<Point> &inlier_parabolaReturn , vector<Point> &inlier_parabolaSoftReturn 
											, vector<Point> &outlier_parabola , Point &vertexParabolaUp , float &dis_thresholdSoft);

void Parabola_Fitting_RANSACDown(const Mat &Src , const vector<Point> &feature_point ,float* &parabola_param
											,  vector<Point> &inlier_parabolaReturn , vector<Point> &inlier_parabolaSoftReturn 
											,  vector<Point> &outlier_parabola , Point &vertexParabolaDown , float &dis_thresholdSoft);



#endif