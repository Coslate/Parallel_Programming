#ifndef PARAMETERS_H_
#define PARAMETERS_H_
#pragma once
#define UINT8 unsigned char
#undef UNICODE
#define _USE_MATH_DEFINES 

#include <math.h>

using namespace std;
using namespace cv;

static const float ang_mul = 180.f/M_PI;
static const float angle_multiply = M_PI/180.f;

static const double PI = std::atan(1.0)*4;
static const int angel_stage2_range = 15;//15//90;//15;//90;//15/*realtime*/;//90;//180;
static const int FRAMEW = 640;	//hysteresis factor for noise reduction
static const int FRAMEH = 480;	//hysteresis factor for noise reduction

static const int MonitorW = 1680;
static const int MonitorH = 1050;

//Limbus feature detection
static const int angle_stage1_start = -90;//-90;//-45;//-90;//-44
static const int angle_stage1_end  = 91;//46;//136;//46
static const int angle_stage2_start = 91;//90;//136;
static const int angle_stage2_end  = 270;//270;//226;//360;//226

static const int inter_angle_stage1_phase1 = 5;
static const int inter_angle_stage1_phase2 = 5;
static const int inter_angle_stage2 = 5;

static const float distanceofFeature = 5;//7//15
static const int radius_stage1_initial = 10;//7//15;
static const int radius_stage2_initial = 10;//7//15;

static const int MAX_WORD_LEN = 300;
static const float gradient_threshold_initial = 2;//15;//20;
static const int number_feature_line = 1;//1;//4;

static const int threshold_center_new_old_distance_terminal = 2;//pixels
static const int threshold_whileloop_centerconvergence_terminal = 10;//after 10 times , if doesn't converge , then it won't converge


//Eyelid feature detection

static const float		EdistanceofFeature = 5;//9;//15;//7//15
static const float      Eradius_initial = 10;//9;//10;//7//15;
static const int			Eupper_angle_range_phase2 = 45;
static const int			Eupper_number_feature_perline_phase2 = 1;//1;//4;
static const int			Eupper_number_PassFeaturePts_perline_phase2 = 0;//1;//4;
static const int			Eupper_inter_angle_phase2 = 5;
static const int			Eupper_numSweep = 1;
static const float		Eupper_gradient_threshold_phase2 = 3;//15;//20;
static const int			Eupper_sourceFeature_inter_range_phase2 = 40;


static const int			Elower_angle_range_phase2 = 45;
static const int			Elower_number_feature_perline_phase2 = 1;//1;//4;
static const int			Elower_number_PassFeaturePts_perline_phase2 = 0;//1;//4;
static const int			Elower_inter_angle_phase2 = 10;
static const int			Elower_numSweep = 1;
static const float		Elower_gradient_threshold_phase2 = 4;//4;//15;//20;
static const int			Elower_sourceFeature_inter_range_phase2 = 40;


//Eyelid feature refinement
static const float determineBigMotiondDistance = 60;


//Iris Boundary Refinement
static const int	 median_filter1DEyeRefinedCenterNonlinearRegion_size = 21;
static const int	 median_filter1DIrisBoundary_size = 5;
static const float iris_GPF_rectSidePercent = 0.7;


//Blink Detection
static const int	 median_filter1D_size = 15;
static const float eyeOpenContourRatioThresh_colorModelBased = 3;
static const float eyeCloseDetermine_irisContourSizeThreshold_percent = 0.05;
static const float eyeRefinedIrisCenter_ExtremeRegion_Right = 17/20.f*FRAMEW;
static const float eyeRefinedIrisCenter_ExtremeRegion_Left = 3/20.f*FRAMEW;
static const float eyeRefinedIrisCenter_ExtremeRegion_Down = 5.5/8.f*FRAMEH;
static const float eyeRefinedIrisCenter_ExtremeRegion_Up = 1.7/8.f*FRAMEH;

//Calibration
static const int median_filter1DEyeRefinedCenter_size = 15;
static const int calUsedW = MonitorW;
static const int calUsedH = MonitorH;

//Eye Gaze Testing
static const int testUsedW = MonitorW;
static const int testUsedH = MonitorH;


//Debug
static const bool showDetail = true;
static const bool printDebug = false;
static bool testVariance = true;
static bool thesisClipImage = false;
static bool thesisClipImageForStarburst = false;
static const int testNum = 0;

#endif