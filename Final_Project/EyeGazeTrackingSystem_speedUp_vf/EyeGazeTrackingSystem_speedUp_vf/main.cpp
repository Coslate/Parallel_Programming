#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc\imgproc_c.h>
#include <opencv2/core/types_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <math.h>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <numeric>
#include <functional>
#include <omp.h>
#include <sys/stat.h> //include for stat
#include <numeric>
#include "Parameters.h"
#include "Limbus_FeatureDetection.h"
#include "Parabola_Fitting_RANSAC.h"
#include "Eyelid_FeatureDetection.h"
#include "BlinkDetection.h"
#include "MappingFunction.h"
#include "TBB_Parallel_SpeedUp.h"
#include <thread>
#include <mutex>
#include <tbb/parallel_for_each.h>
#include <tbb/task_scheduler_init.h>
//#include <tbb/tbb.h>



using namespace std;
using namespace cv;


//Video writer output
//#define writeResult 0;
bool outputAvgFps = true;

//Multithread
int thread_num = 1;

//Testing File
char testVarOutFileName[MAX_WORD_LEN];
char analysisGazeOutputDir [MAX_WORD_LEN]= ".\\Result\\Gaze_Result_Polynomial";
char testVariancePtsDir[MAX_WORD_LEN] = ".\\Test_Mean_Variance_Data";
char testVarianceFolder[MAX_WORD_LEN] = ".\\pp_test_med_down_15";
char imageTestScene[MAX_WORD_LEN] = ".\\testImage\\image_test_scene.jpg";
char imageTestScene_words[MAX_WORD_LEN] = ".\\testImage\\image_testWords_scene.jpg";


/*TestSubject*/
bool testSubject = false;
char subjectFileName[MAX_WORD_LEN];
char calibrationNumTimesName[MAX_WORD_LEN];


/*Calibration*/
bool orderInput = false;
int calibrationInterTimePoints = 3;//time between calibration points is 5 secs
int calibrationInterMappingFunctionOrder = 2;
int calPtsLength = 5;//Use calPtsLength pts for calibration  chessboard 3X3 ~ 8X8
Mat EyePtsTransformMat_Opt;
Mat ScenePtsTransformMat_Opt;
double time_cal;
bool isChessBoardSideOdd = false;
bool calibrationProcedureBegin = false;
bool getCalibrationPoints = true;
bool getSlicingMap = false;
int count_calProcedure = 0;
int count_times_2secs = 0;
int posLine_y = 0;
int time_duringLast = 0;
int countCalibrationPtsMoveTimes = 0;
int test_count_cal = 0;
Point filteredEyeRefinedCenter(0 , 0);
deque<int> posQueueX;
deque<int> posQueueY;
std::vector<Point> calibratedEyeRefinedCenter;
std::vector<Point> calibratedCalPoints;
std::vector<Point> calInterAnchorPoints;
std::vector<sliceMapElement> mappingSliceMap;
std::vector<cv::Point> calibrationPatternPtsStep_One;
bool calibrationProcedureDone = false;
Point calibratedChessBoardPtsLast(0 , 0);
int count_times_2secsNext = 0;
int countEyePut = 0;
int countCalibrationPatternPtsStep_One_pos = 0;
int orderOpt;
int calibrationPts_space = calibrationPattern::Step_space_two;
int calBrationMethod = calibrationMethod::Polynomial_All;
int video_num = 15;

/*Profiling*/
double time_start   = 0;
double time_end     = 0;
double time_start_in_function = 0;
double time_end_in_function   = 0;
bool   do_profiling = true;
bool   do_profiling_white_balance = false;
double avg_time     = 0;
double t            = 0;
vector<double> time_image_preprocessing_white_balance;
vector<double> time_image_preprocessing_cvt_color;
vector<double> time_image_preprocessing_hist_equal;
vector<double> time_image_preprocessing_high_freq_removal;
vector<double> time_iris_model_calculate_h_s_histogram;
vector<double> time_iris_model_valid_testing;
vector<double> time_eye_position_detection;
//vector<int64> time_eye_position_detection_high_freq_removal;
vector<double> time_eye_position_detection_coarse_iris_location_color;
vector<double> time_eye_position_detection_coarse_iris_location_color_serial;
vector<double> time_eye_position_detection_coarse_iris_location_cdf;
vector<double> time_eye_position_detection_coarse_iris_location_cdf_serial;
vector<double> time_eye_position_detection_iris_region_extraction;
vector<double> time_eye_position_detection_eyelid_feature_detection;
vector<double> time_eye_position_detection_parabola_model_fitting;
vector<double> time_eye_position_detection_limbus_feature_detection;
vector<double> time_eye_position_detection_limbus_feature_detection_serial;
vector<double> time_eye_position_detection_center_calculation;
vector<double> time_eye_position_detection_refresh_iris_min_area_region;
vector<double> time_eye_position_detection_refresh_min_area_region;
vector<double> time_eye_position_detection_refresh_iris_ROI_region;
vector<double> time_eye_position_test;
vector<double> time_eye_position_detection_eyelid_feature_detection_district1;
vector<double> time_eye_position_detection_eyelid_feature_detection_district2;
vector<double> time_eye_position_detection_eyelid_feature_detection_district3;
vector<double> time_eye_position_detection_eyelid_feature_detection_district4;
vector<double> time_eye_position_detection_eyelid_feature_detection_district4_0;
vector<double> time_eye_position_detection_eyelid_feature_detection_district4_1;
vector<double> time_eye_position_detection_eyelid_feature_detection_district4_1_1;
vector<double> time_eye_position_detection_eyelid_feature_detection_district4_1_2;
vector<double> time_eye_position_detection_eyelid_feature_detection_district4_1_3;
vector<double> time_eye_position_detection_eyelid_feature_detection_district4_1_4;
vector<double> time_eye_position_detection_eyelid_feature_detection_district4_1_5;
vector<double> time_eye_position_detection_eyelid_feature_detection_district4_2;
//vector<int64> time_eye_position_detection_eyelid_feature_detection_district4_2_core;
vector<double> time_eye_position_detection_eyelid_feature_detection_district4_3;
vector<double> time_eye_position_detection_eyelid_feature_detection_district4_3_1;
vector<double> time_eye_position_detection_eyelid_feature_detection_district4_3_2;
vector<double> time_eye_position_detection_eyelid_feature_detection_district4_3_3;
vector<double> time_eye_position_detection_eyelid_feature_detection_district4_3_4;
vector<double> time_eye_position_detection_eyelid_feature_detection_district4_4;
vector<double> time_eye_position_detection_eyelid_feature_detection_district4_5;
vector<double> time_eye_position_detection_eyelid_feature_detection_district4_local_sum;
vector<double> time_eye_position_detection_eyelid_feature_detection_district5;
vector<double> time_eye_position_detection_eyelid_feature_detection_district5_core;
vector<double> time_eye_position_detection_eyelid_feature_detection_district5_merge;

/*Eye Gaze Testing*/
int testNumOfPts = 16;
int testInterTimePoints = 5;//time between testing points is 5 secs
float distanceOfMonitor = 50;//Distance to caculate accuracy in degree
float pixelToCentiMeter = 0.028;//Due to the spec of monitor , transform the pixel to centimeter
vector<Point> testPtsVec;
vector<GazeTestingElement> gazeTesting_vec;
Point dispTestPts(0 , 0);
Point dispTestPtsLast(-1 , -1);
double time_duringTest;
double time_duringCalLast = -1;
int* rand_numTestPts;
bool eyeGazeTestProcedure = false;
int countInEyeGazeTestingFunction = 0;
int countGazePut = 0;
bool realGazeInput = false;

/*Eye Corner Tracking*/
int roi_LeftWidth     = 100;
int roi_LeftHeight	 = 80;
int roi_RightWidth	 = 100;
int roi_RightHeight	 = 80;
Point rectLeftCornerStartPoint;
Point rectRightCornerStartPoint;
Rect region_of_interestLeftCorner ;
Rect region_of_interestRightCorner; 
Mat ROI_Leftcorner ;
Mat ROI_Rightcorner ;
Point leftCornerOriginalPoint = Point(10,281);
Point rightCornerOriginalPoint = Point(598,281);		
bool setEyeCornerAndEyePosReady = true;
bool doNotdisplayEyeCorner = false;


int leftCorrect_x_left ,leftCorrect_x_right , leftCorrect_y_up , leftCorrect_y_down;
int rightCorrect_x_left , rightCorrect_x_right , rightCorrect_y_up , rightCorrect_y_down;
int width_LeftROIforFollowingTM ,height_LeftROIforFollowingTM;
int width_RightROIforFollowingTM ,height_RightROIforFollowingTM;


/*Big Motion*/
Point eyeCoarseCenterLast(0 , 0);
Point eyeRefinedIrisCenterLast(0 , 0);
deque<int> iris_boundaty_Left;
deque<int> iris_boundaty_Right;
deque<int> posNonlinearRegionEyeQueueY;
float meanLastIrisCenterInterFrameFilter = 0;


/*Refined Iris Center*/
std::vector<Point2f> irisCenterEstimation;
float eyeCoarseIrisCenter_ExtremeRegion_Right = 15/20.f*FRAMEW;
float eyeCoarseIrisCenter_ExtremeRegion_Down = 5.5/8.f*FRAMEH;
float eyeCoarseIrisCenter_ExtremeRegion_Up = 2.5/8.f*FRAMEH;
float eyeCoarseIrisCenter_ExtremeRegion_Left = 5/20.f*FRAMEW;
float eyeCenter_rightestPos = 15.5/20.f*FRAMEW;
float eyeCenter_lowestPos   = 5/8.f*FRAMEH;
float eyeCenter_uppestPos  = 3.5/8.f*FRAMEH;
float eyeCenter_leftestPos = 5/20.f*FRAMEW;
Point eyeRefinedIrisCenter(0 , 0);	
int jitterUppest_y = FRAMEH-1;
int jitterLowest_y = 0;




/*Blink Detection*/
deque<float> ratio_queue;			//for median filter of ratio 
bool eyeState = Eyeopen;
bool pre_eyeState;
int voluntaryBlinkCount = 0;
int non_voluntaryBlinkCount = 0;
int countCloseFrame = 0;
float eyeCloseDetermine_irisContourSizeThreshold_colorModelBased;
Point vertexParabolaUpperLast;
Point vertexParabolaLowerLast;
Point vertexParabolaUpperFirstFrame;
Point vertexParabolaLowerFirstFrame;
bool turn_on_eye_blink_detection = false;




/*Iris Model*/
// Quantize the hue to 30 levels and the saturation to 32 levels
int Iris_hbins = 5, Iris_sbins = 5;
// hue varies from 0 to 179, see cvtColor
float hranges[] = { 0, 180 };
// saturation varies from 0 (black-gray-white) to 255 (pure spectrum color)
float sranges[] = { 0, 256 };

const float* ranges[] = { hranges, sranges};
// we compute the histogram from the 0-th and 1-st channels
int channels [] = {0, 1};
int Iris_histSize[] = {Iris_hbins, Iris_sbins};
int IrishValuePerBin = (hranges[1] - hranges[0])/(float)Iris_hbins;
int IrissValuePerBin = (sranges[1] - sranges[0])/(float)Iris_sbins;
const int Iris_PerBin[] = { IrishValuePerBin, IrissValuePerBin};
bool gotIrisROI = false;
bool selectObjectIris = false;
bool irisDynamicMaskGeneration = false;
bool writeIrisModel = false;
bool readIrisModel = false;
Point origin_iris_ROI;
Rect selection;
std::vector<Mat>  Iris_hist_vector;
Mat Iris_hist;
Mat Iris_ROI_forModel;
Mat Iris_Mask = Mat::zeros(FRAMEH , FRAMEW , CV_8UC1);
char IrisHisto_output_fileName[MAX_WORD_LEN];
bool caculateIris_Mask_done = false;
int countRefreshTimes = 0;
int iris_maskModel_refreshFrame;





/*Refresh Iris Hist Model*/
int morphological_erode_refinedEyeConvexHull_size = 45;
int morphological_erode_IrisRegionValidTesting_size = 13;
float irisRate_max = -FLT_MAX;
int iris_maskModel_refreshFrame_Initial = 1;
int iris_maskModel_refreshFrame_AfterGet = 100;
float iris_colorModelValidTestingIrisRate_initial = 6;//40
float iris_colorModelIrisRate_pixelInOthersOne = 0.5;// IrisRegionValidTestSize*45%


/*Iris Mask For Limbus */
const float intensityErosionBaseSize_experiment = 43067;
const float colorErosionBaseSize_experiment = 48248;
const float colorUpErosionBaseSize_experiment = 38381;
const float eyeRefinedConvexForIrisMaskColorModelBaseSize_experiment = 45433.6;
int irisMaskForLimbusErode_sizeUp = 25;
int irisMaskForLimbusErode_size = 23;



/*Set Eye Corner*/
std::vector<Point>eyeCorners;
static bool setCornerPoint = false;
static const int countSetCornerPoint = 2;
static int CurrentCornerPointPos = 0;


/*Sortimg 1D Data of General Projection Function*/
class FindPeakof1D_Data{
public:
	FindPeakof1D_Data(){}
	int number;
	int element;	

	bool operator<(const FindPeakof1D_Data&); // use global function
};

bool FindPeakof1D_Data::operator<(const FindPeakof1D_Data& src){
	return (this->element<src.element?true:false);
}


/*Gaze Estimation*/
Point gazePoint(0 , 0);
double *mapping_paramOptX;
double *mapping_paramOptY;
int numberOfVar;
double meanSquareError;


/*GUI*/
string enterstring;
int changeScene = 0;


Mat Scene_image;
Mat Scene_calibration = Mat::zeros(FRAMEH , FRAMEW , CV_8UC3);
Mat Scene_chessboard = Mat::zeros(calUsedH , calUsedW , CV_8UC3);
Mat Scene_gazetest;
Mat3b Frame;
Mat3b Frame_wh = Mat::zeros(FRAMEH , FRAMEW , CV_8UC3);
Mat3b RGB_Normalised;
Mat EyeImageForTestingIrisHistModel;
Mat IrisRegionValidTesting;
Mat Histogram_Eq = Mat::zeros(FRAMEH , FRAMEW , CV_8UC1);
Mat Histogram_Eq2 = Mat::zeros(FRAMEH , FRAMEW , CV_8UC1);
Mat Frame_Gray = Mat::zeros(FRAMEH , FRAMEW , CV_8UC1);
Mat Frame_Gray_Log;
Mat Edgefield;
Mat ValleyPeakfield;
Mat Cornerfield;
Mat Cornerfield_Log;
Mat EyePosition_Result = Mat::zeros(FRAMEH , FRAMEW , CV_8UC3);
Mat EyePosition_CenterResult = Mat::zeros(FRAMEH , FRAMEW , CV_8UC3);
Mat PrevGray;
Mat Frame_HSV;
Mat HSV[3];
Mat Frame_colorReduce;



//Testing Perfoemance
std::vector<Point> groundTruth;
std::vector<Point> groundTruthWriteOut;

//--testing variance--//
std::vector<Point2f> centerEstimation_coarseCenter;
std::vector<Point2f> centerEstimation_ellipseFineCenter;
std::vector<Point2f> centerEstimation_convexHullFineCenter;
std::vector<float> ellipseArea;
std::vector<float> convexHullArea;


#pragma region
inline void Draw_Cross(Mat &image, int centerx, int centery, int x_cross_length, int y_cross_length, Scalar &color)
{
  Point pt1,pt2,pt3,pt4;

  pt1.x = centerx - x_cross_length;
  pt1.y = centery;
  pt2.x = centerx + x_cross_length;
  pt2.y = centery;

  pt3.x = centerx;
  pt3.y = centery - y_cross_length;
  pt4.x = centerx;
  pt4.y = centery + y_cross_length;

  line(image,pt1,pt2,color,1,8);
  line(image,pt3,pt4,color,1,8);
}


//inline double remap(uchar &v, const double &min, const double &max){
//	return (v-min)/(double)(max-min);
//}


inline bool checkpoint(const int &x ,const int &y , const Mat &src){
	if(x<0||x>=src.cols||y<0||y>=src.rows)
		return false;
	else
		return true;
}

inline bool checkpoint(const Point &pt ,  const Mat &src){
	if(pt.x<0||pt.x>=src.cols||pt.y<0||pt.y>=src.rows)
		return false;
	else
		return true;
}

inline bool checkpointinROI(const int x ,const int y , const Point &UpLeftPt , const int WidthOfROP , const int HwightOfROP){
	if(x<UpLeftPt.x || x>UpLeftPt.x+WidthOfROP || y<UpLeftPt.y || y>UpLeftPt.y + HwightOfROP)
		return false;
	else
		return true;
}
inline void ImageInverse(const Mat& src , Mat& dst){	
	for(int i=0;i<src.rows;++i){
		for(int j=0;j<src.cols;++j){
			dst.at<uchar>(i , j) = 255 - src.at<uchar>(i , j);
		}
	}
}


inline const float& DistanceCaculateEuclidean(const Point2f &x1 , const Point2f &x2){
	Point2f vectorLineX1X2(x2.x - x1.x , x2.y - x1.y);
	return sqrtf(vectorLineX1X2.x*vectorLineX1X2.x + vectorLineX1X2.y*vectorLineX1X2.y);
}

inline void Parallel_MinMax_Process1(const Mat &src1, double &min_1, double &max_1, const int remain_threads) {
	//minMaxLoc(src1, &min_1, &max_1, NULL, NULL);

	if (remain_threads > 0) {
		//Mat min_val_cand = Mat::zeros(1, remain_threads, CV_64F);
		//Mat max_val_cand = Mat::zeros(1, remain_threads, CV_64F);
		double *min_val_cand = new double[remain_threads]();
		double *max_val_cand = new double[remain_threads]();
		min_1 = FLT_MAX;
		max_1 = -FLT_MAX;

		cv::parallel_for_(cv::Range(0, remain_threads), Parallel_process_find_min_max_arr(src1, min_val_cand, max_val_cand, remain_threads));
		//cv::parallel_for_(cv::Range(0, remain_threads), Parallel_process_find_min_max(src1, min_val_cand1, max_val_cand1, remain_threads));
		//minMaxLoc(min_val_cand, &min_1, NULL, NULL, NULL);
		//minMaxLoc(max_val_cand, NULL, &max_1, NULL, NULL);
		for (int i = 0; i < remain_threads; ++i) {
			if (min_1 > min_val_cand[i]) {
				min_1 = min_val_cand[i];
			}
			if (max_1 < max_val_cand[i]) {
				max_1 = max_val_cand[i];
			}
		}

		delete [] min_val_cand;
		delete [] max_val_cand;
	}else {
		minMaxLoc(src1, &min_1, &max_1, NULL, NULL);
	}
}

inline void Parallel_MinMax_Process2(const Mat &src1, const Mat&src2, double &min_1, double &max_1, double &min_2, double &max_2, const int remain_threads) {
	//minMaxLoc(src1, &min_1, &max_1, NULL, NULL);
	//minMaxLoc(src2, &min_2, &max_2, NULL, NULL);

	if (remain_threads > 0) {
		//Mat min_val_cand1 = Mat::zeros(1, remain_threads, CV_64F);
		//Mat max_val_cand1 = Mat::zeros(1, remain_threads, CV_64F);
		//Mat min_val_cand2 = Mat::zeros(1, remain_threads, CV_64F);
		//Mat max_val_cand2 = Mat::zeros(1, remain_threads, CV_64F);
		double *min_val_cand1 = new double[remain_threads]();
		double *max_val_cand1 = new double[remain_threads]();
		double *min_val_cand2 = new double[remain_threads]();
		double *max_val_cand2 = new double[remain_threads]();
		min_1 = min_2 = FLT_MAX;
		max_1 = max_2 = -FLT_MAX;

		//cv::parallel_for_(cv::Range(0, remain_threads), Parallel_process_find_min_max(src1, min_val_cand1, max_val_cand1, remain_threads));
		//minMaxLoc(min_val_cand1, &min_1, NULL, NULL, NULL);
		//minMaxLoc(max_val_cand1, NULL, &max_1, NULL, NULL);

		//cv::parallel_for_(cv::Range(0, remain_threads), Parallel_process_find_min_max(src2, min_val_cand2, max_val_cand2, remain_threads));
		//minMaxLoc(min_val_cand2, &min_2, NULL, NULL, NULL);
		//minMaxLoc(max_val_cand2, NULL, &max_2, NULL, NULL);

		cv::parallel_for_(cv::Range(0, remain_threads), Parallel_process_find_min_max_arr(src1, min_val_cand1, max_val_cand1, remain_threads));
		cv::parallel_for_(cv::Range(0, remain_threads), Parallel_process_find_min_max_arr(src2, min_val_cand2, max_val_cand2, remain_threads));

		for (int i = 0; i < remain_threads; ++i) {
			if (min_1 > min_val_cand1[i]) {
				min_1 = min_val_cand1[i];
			}
			if (max_1 < max_val_cand1[i]) {
				max_1 = max_val_cand1[i];
			}
			if (min_2 > min_val_cand2[i]) {
				min_2 = min_val_cand2[i];
			}
			if (max_2 < max_val_cand2[i]) {
				max_2 = max_val_cand2[i];
			}
		}
		delete [] min_val_cand1;
		delete [] max_val_cand1;
		delete [] min_val_cand2;
		delete [] max_val_cand2;
	}
	else {
		minMaxLoc(src1, &min_1, &max_1, NULL, NULL);
		minMaxLoc(src2, &min_2, &max_2, NULL, NULL);
	}
}

inline void Parallel_MinMax_Process3(const Mat &src1, const Mat&src2, const Mat&src3, double &min_1, double &max_1, double &min_2, double &max_2, double &min_3, double &max_3, const int remain_threads) {
	//minMaxLoc(src1, &min_1, &max_1, NULL, NULL);
	//minMaxLoc(src2, &min_2, &max_2, NULL, NULL);
	//minMaxLoc(src3, &min_3, &max_3, NULL, NULL);

	if (remain_threads > 0) {
		//Mat min_val_cand1 = Mat::zeros(1, remain_threads, CV_64F);
		//Mat max_val_cand1 = Mat::zeros(1, remain_threads, CV_64F);
		//Mat min_val_cand2 = Mat::zeros(1, remain_threads, CV_64F);
		//Mat max_val_cand2 = Mat::zeros(1, remain_threads, CV_64F);
		double *min_val_cand1 = new double[remain_threads]();
		double *max_val_cand1 = new double[remain_threads]();
		double *min_val_cand2 = new double[remain_threads]();
		double *max_val_cand2 = new double[remain_threads]();
		double *min_val_cand3 = new double[remain_threads]();
		double *max_val_cand3 = new double[remain_threads]();
		min_1 = min_2 = min_3 = FLT_MAX;
		max_1 = max_2 = max_3 = -FLT_MAX;


		//cv::parallel_for_(cv::Range(0, remain_threads), Parallel_process_find_min_max(src1, min_val_cand1, max_val_cand1, remain_threads));
		//minMaxLoc(min_val_cand1, &min_1, NULL, NULL, NULL);
		//minMaxLoc(max_val_cand1, NULL, &max_1, NULL, NULL);

		//cv::parallel_for_(cv::Range(0, remain_threads), Parallel_process_find_min_max(src2, min_val_cand2, max_val_cand2, remain_threads));
		//minMaxLoc(min_val_cand2, &min_2, NULL, NULL, NULL);
		//minMaxLoc(max_val_cand2, NULL, &max_2, NULL, NULL);

		cv::parallel_for_(cv::Range(0, remain_threads), Parallel_process_find_min_max_arr(src1, min_val_cand1, max_val_cand1, remain_threads));
		cv::parallel_for_(cv::Range(0, remain_threads), Parallel_process_find_min_max_arr(src2, min_val_cand2, max_val_cand2, remain_threads));
		cv::parallel_for_(cv::Range(0, remain_threads), Parallel_process_find_min_max_arr(src3, min_val_cand3, max_val_cand3, remain_threads));

		for (int i = 0; i < remain_threads; ++i) {
			if (min_1 > min_val_cand1[i]) {
				min_1 = min_val_cand1[i];
			}
			if (max_1 < max_val_cand1[i]) {
				max_1 = max_val_cand1[i];
			}
			if (min_2 > min_val_cand2[i]) {
				min_2 = min_val_cand2[i];
			}
			if (max_2 < max_val_cand2[i]) {
				max_2 = max_val_cand2[i];
			}
			if (min_3 > min_val_cand3[i]) {
				min_3 = min_val_cand3[i];
			}
			if (max_3 < max_val_cand3[i]) {
				max_3 = max_val_cand3[i];
			}
		}
		delete [] min_val_cand1;
		delete [] max_val_cand1;
		delete [] min_val_cand2;
		delete [] max_val_cand2;
		delete [] min_val_cand3;
		delete [] max_val_cand3;
	}
	else {
		minMaxLoc(src1, &min_1, &max_1, NULL, NULL);
		minMaxLoc(src2, &min_2, &max_2, NULL, NULL);
		minMaxLoc(src3, &min_3, &max_3, NULL, NULL);
	}
}

inline void Parallel_MinMax_Process3_ver2(const Mat3b &src, Mat3b &out, const Mat &src1, const Mat&src2, const Mat&src3, const int remain_threads) {
	double *min_val_cand1 = new double[remain_threads]();
	double *max_val_cand1 = new double[remain_threads]();
	double *min_val_cand2 = new double[remain_threads]();
	double *max_val_cand2 = new double[remain_threads]();
	double *min_val_cand3 = new double[remain_threads]();
	double *max_val_cand3 = new double[remain_threads]();
	double MIN_b = FLT_MAX;
	double MIN_g = FLT_MAX;
	double MIN_r = FLT_MAX;
	double MAX_b = -FLT_MAX;
	double MAX_g = -FLT_MAX;
	double MAX_r = -FLT_MAX;


	/*FIND MAX PIXEL VALUE*/
	cv::parallel_for_(cv::Range(0, remain_threads), Parallel_process3_find_min_max_arr(src1, src2, src3, min_val_cand1, max_val_cand1, min_val_cand2, max_val_cand2, min_val_cand3, max_val_cand3, remain_threads));

	//double time_start1 = getTickCount();
	for (int i = 0; i < remain_threads; ++i) {
		if (MIN_b > min_val_cand1[i]) {
			MIN_b = min_val_cand1[i];
		}
		if (MAX_b < max_val_cand1[i]) {
			MAX_b = max_val_cand1[i];
		}
		if (MIN_g > min_val_cand2[i]) {
			MIN_g = min_val_cand2[i];
		}
		if (MAX_g < max_val_cand2[i]) {
			MAX_g = max_val_cand2[i];
		}
		if (MIN_r > min_val_cand3[i]) {
			MIN_r = min_val_cand3[i];
		}
		if (MAX_r < max_val_cand3[i]) {
			MAX_r = max_val_cand3[i];
		}
	}
	//std::cout << "white_serial = " << (getTickCount() - time_start1) / getTickFrequency() << std::endl;

	delete [] min_val_cand1;
	delete [] max_val_cand1;
	delete [] min_val_cand2;
	delete [] max_val_cand2;
	delete [] min_val_cand3;
	delete [] max_val_cand3;

	//std::cout << "=================Parallel : ===============" << std::endl;
	//std::cout << "min_b = " << MIN_b << ", max_b = " << MAX_b << std::endl;
	//std::cout << "min_g = " << MIN_g << ", max_g = " << MAX_g << std::endl;
	//std::cout << "min_r = " << MIN_r << ", max_r = " << MAX_r << std::endl;

	/*Remapping*/
	cv::parallel_for_(cv::Range(0, remain_threads), Parallel_process3_remap(src, out, MIN_b, MAX_b, MIN_g, MAX_g, MIN_r, MAX_r, remain_threads));
}

inline void Parallel_Component_Stretching(const Mat3b &src, Mat3b &dst, const int thread_num) {
	dst = Mat::zeros(src.rows, src.cols, CV_8UC3);

	double MAX_r;
	double MAX_g;
	double MAX_b;

	double MIN_r;
	double MIN_g;
	double MIN_b;

	Point minLoc_r;
	Point minLoc_g;
	Point minLoc_b;

	Point maxLoc_r;
	Point maxLoc_g;
	Point maxLoc_b;

	/*SPLIT SRC TO RGB CHANNELS*/
	vector<Mat> bgr_planes;
	vector<thread> threads;
	split(src, bgr_planes);

	double time_loc_start;
	double time_loc_end;

	time_loc_start = getTickCount();
	/*FIND MAX PIXEL VALUE*//*Remapping*/
	Parallel_MinMax_Process3_ver2(src, dst, bgr_planes[0], bgr_planes[1], bgr_planes[2], thread_num);

	//std::cout << "=================Parallel : ===============" << std::endl;
	//std::cout << "min_b = " << MIN_b << ", max_b = " << MAX_b << std::endl;
	//std::cout << "min_g = " << MIN_g << ", max_g = " << MAX_g << std::endl;
	//std::cout << "min_r = " << MIN_r << ", max_r = " << MAX_r << std::endl;
	time_loc_end = getTickCount();
	time_image_preprocessing_white_balance.push_back(time_loc_end - time_loc_start);
}

inline void Component_Stretching(const Mat3b &src , Mat3b &dst){
	dst = Mat::zeros(src.rows , src.cols , CV_8UC3);
	
	double MAX_r;
	double MAX_g;
	double MAX_b;

	double MIN_r;
	double MIN_g;
	double MIN_b;

    Point minLoc_r; 
	Point minLoc_g; 
	Point minLoc_b;

	Point maxLoc_r;
	Point maxLoc_g;
	Point maxLoc_b;

	/*SPLIT SRC TO RGB CHANNELS*/
	vector<Mat> bgr_planes;
	split( src, bgr_planes );
	
	/*FIND MAX PIXEL VALUE*/
	//double time_start_loc;
	//double time_end_loc;

	//if (do_profiling || do_profiling_white_balance) {
	//	time_start_loc = getTickCount();
	//}

    minMaxLoc( bgr_planes[0], &MIN_b, &MAX_b, &minLoc_b, &maxLoc_b);
	minMaxLoc( bgr_planes[1], &MIN_g, &MAX_g, &minLoc_g, &maxLoc_g );
	minMaxLoc( bgr_planes[2], &MIN_r, &MAX_r, &minLoc_r, &maxLoc_r );

	//std::cout << "=================Serial : ===============" << std::endl;
	//std::cout << "min_b = " << MIN_b << ", max_b = " << MAX_b << std::endl;
	//std::cout << "min_g = " << MIN_g << ", max_g = " << MAX_g << std::endl;
	//std::cout << "min_r = " << MIN_r << ", max_r = " << MAX_r << std::endl;

	cv::Mat_<cv::Vec3b>::const_iterator it = src.begin();
	cv::Mat_<cv::Vec3b>::const_iterator itend = src.end();
	cv::Mat_<cv::Vec3b>::iterator itout = dst.begin();

	for(;it!=itend;++it , ++itout){
		Vec3b vi = *it;

		double R_new;
		double G_new;
		double B_new;

		R_new = remap(vi.val[2] , MIN_r ,MAX_r);
		G_new = remap(vi.val[1] , MIN_g ,MAX_g);
		B_new = remap(vi.val[0] , MIN_b ,MAX_b);		

		cv::Vec3b vout = *it;

		vout.val[0] = B_new*255;
		vout.val[1] = G_new*255;
		vout.val[2] = R_new*255;

		*itout = vout;
	}

	//if (do_profiling || do_profiling_white_balance) {
	//	time_end_loc = getTickCount();
	//	time_white_balance_serial.push_back(time_end_loc - time_start_loc);
	//}
}

inline void Morphology_Operations(const Mat &src , Mat &dst , const int &operation, const int &morph_size ,  const int &morph_elem = 0){  
  Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

  /// Apply the specified morphology operation
  morphologyEx( src, dst, operation, element );  
 }

static bool sort_using_greater_thany(const Point &u,const Point &v){
	  return u.y > v.y;
 }


inline void CalcHistogram(const Mat &src , Mat &dst/*float type*/){
	/// Establish the number of bins
	Mat dst_norm;
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 } ;
	const float* histRange = { range };
	calcHist( &src, 1, 0, Mat(), dst, 1, &histSize, &histRange);
}

inline void Log_Trans(const Mat&src , Mat &dst , float c){
	const float mulConstant = 255*c/log(255.0);
	float table[256];
	for (int i = 0; i < 256; ++i) {
		table[i] = cvRound(logf(1.0f+i)*mulConstant);
	}
	dst = Mat::zeros(src.rows , src.cols ,CV_32FC1);
	//#pragma omp parallel for
	for(int i=0;i<src.rows;++i){
		for(int j=0;j<src.cols;++j){			
			dst.at<float>(i , j)=table[src.at<uchar>(i , j)];
		}							
	}	
}
inline void ConvexHullComp(const vector<Point> &histo_points , vector<Point> &hull_pt){
	bool finish = false;
	int index = 0;
	int index_renew;
	hull_pt.push_back(histo_points[0]);
	while(!finish){
		float slope_max = -FLT_MAX;
		for(int i=index+1;i<histo_points.size();++i){
			float slope = -(histo_points[i].y - histo_points[index].y)/(float)(histo_points[i].x - histo_points[index].x);			
			if(slope>=slope_max){				
				index_renew = i;
				slope_max = slope;
			}else{
				continue;
			}
		}//end for
		hull_pt.push_back(histo_points[index_renew]);
		if(index_renew==histo_points.size()-1){
			finish=true;
		}else{
			index = index_renew;
		}	
	}//end while
}
inline float LineFunction(const Point &pt1 , const Point &pt2 , const int pt_x){
	float slope = (pt2.y - pt1.y)/(float)(pt2.x - pt1.x);
	return ((float)pt1.y+slope*(float)(pt_x - pt1.x));
}

inline void YESTransform(const Mat3b &src , Mat &dstY , Mat &dstE , Mat &dstS){
	//Split src to RGB channels	
	vector<Mat> bgr_planes;
	split( src, bgr_planes );
	Mat tmp;


	//Y	
	addWeighted( bgr_planes[0], 0.063, bgr_planes[1], 0.684, 0, tmp );
	addWeighted( bgr_planes[2], 0.253, tmp , 1, 0, dstY );	
	//E	
	addWeighted( bgr_planes[1], -0.5, bgr_planes[2], 0.5, 0, dstE );
	//S
	addWeighted( bgr_planes[0], -0.5, bgr_planes[1], 0.250, 0, tmp );
	addWeighted( bgr_planes[2], 0.250, tmp , 1, 0, dstS );		
}

inline bool FindMAXConnextedComponent(const Mat& Src , vector<Point> &max_contours , Mat &Contour_dst){
	Contour_dst = Mat::zeros(Src.rows , Src.cols , CV_8UC1);
	Mat Contour_src = Src.clone();
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	

	findContours( Contour_src, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );
	
	if(contours.size()==0){
		printf("There is no contours in FindMAXConnextedComponent()\n");
		return false;		
	}

	if( !contours.empty() && !hierarchy.empty() ){
		int idx = 0;
		float max_contour_size = -FLT_MAX;		
		
		for(int i=0;i<contours.size();++i){
			if(contours[i].size()>max_contour_size){
				max_contour_size = contours[i].size();				
				idx = i;
			}
		}
		max_contours = contours[idx];				
		drawContours( Contour_dst, contours, idx, Scalar(255 , 255 , 255), cv::FILLED, 8, hierarchy );		
	}	
	return true;
}

inline void FindALLContours(const Mat& Src  , /*Mat &Contour_dst ,*/ vector<vector<Point> > &contours){	
	Mat Contour_src = Src.clone();	
	vector<Vec4i> hierarchy;

	findContours( Contour_src, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );
}


inline void RemoveSamllConnectedComponent(const Mat& src 
																	, vector<Point>& OutputConnectedPoints 
																	, Mat& OutMask 
																	, float OffFactor)
{
	OutMask = Mat::zeros(src.rows , src.cols , CV_8UC1);
	Mat contour_src = src.clone();
	vector<vector<Point> > contours;
	vector<Point> max_contours;
	vector<Vec4i> hierarchy;

	findContours( contour_src, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );
	
	if( !contours.empty() && !hierarchy.empty() ){
		// iterate through all the top-level contours,
		// draw each connected component with its own random color
		
		//Find Max contours
		int idx = 0;
		float max_contour_size = -FLT_MAX;				
		for(int i=0;i<contours.size();++i){
			if(contours[i].size()>max_contour_size){
				max_contour_size = contours[i].size();				
				idx = i;
			}
		}
		max_contours = contours[idx];		
		idx = 0;

		//Select contours except those smaller than max_countours*OffFactor
		for( ; idx >= 0; idx = hierarchy[idx][0] ){
			if(contours[idx].size()<max_contours.size()*OffFactor)
				continue;


			Scalar color( 255 , 255 , 255);
			drawContours( OutMask, contours, idx, color, cv::FILLED, 8, hierarchy );
		}
		for(int i=0;i<OutMask.rows;++i){
			for(int j=0;j<OutMask.cols;++j){
				if(OutMask.at<uchar>(i , j) == 255){
					OutputConnectedPoints.push_back(Point(j , i));
				}
			}
		}

/*		float max_contour_size = -FLT_MAX;		
		
		for(int i=0;i<contours.size();++i){
			if(contours[i].size()>max_contour_size){
				max_contour_size = contours[i].size();				
				idx = i;
			}
		}
		max_contours = contours[idx];		
		
		drawContours( contour_dst, contours, idx, Scalar(255 , 255 , 255), cv::FILLED, 8, hierarchy );	*/	
	}	
}


inline unsigned char normalize_harris(const double &min , const double &max , const float in){
	return (255*1/(max-min)*in + 255*(min/(min-max)));
}

inline void Norm(const Mat &src , Mat &dst){
	double min , max;
	minMaxLoc( src, &min, &max, 0, 0, Mat() );	
	for(int i=0;i<src.rows;++i){
		for(int j=0;j<src.cols;++j){
			dst.at<uchar>(i , j) = normalize_harris(min , max , src.at<float>(i , j));
		}
	}
}




inline double asinh(double value){   
	double returned;

	if(value>0)
		returned = log(value + sqrt(value * value + 1));
	else
		returned = -log(-value + sqrt(value * value + 1));

	return(returned);
}





inline void Draw_Cross(Mat3b &image, int centerx, int centery, int x_cross_length, int y_cross_length, Scalar &color)
{
  Point pt1,pt2,pt3,pt4;

  pt1.x = centerx - x_cross_length;
  pt1.y = centery;
  pt2.x = centerx + x_cross_length;
  pt2.y = centery;

  pt3.x = centerx;
  pt3.y = centery - y_cross_length;
  pt4.x = centerx;
  pt4.y = centery + y_cross_length;

  line(image,pt1,pt2,color,1,8);
  line(image,pt3,pt4,color,1,8);
}

inline void Draw_Cross(Mat &image, const int &centerx, const int &centery, const int &x_cross_length, const int &y_cross_length
									, const Scalar &color ,const int &thick)
{
  Point pt1,pt2,pt3,pt4;

  pt1.x = centerx - x_cross_length;
  pt1.y = centery;
  pt2.x = centerx + x_cross_length;
  pt2.y = centery;

  pt3.x = centerx;
  pt3.y = centery - y_cross_length;
  pt4.x = centerx;
  pt4.y = centery + y_cross_length;

  line(image,pt1,pt2,color,thick,8);
  line(image,pt3,pt4,color,thick,8);
}

inline void Draw_Cross(Mat &dst , const vector<Point> &feature , int feature_index , bool is_inliers){
	if(is_inliers){
		Draw_Cross(dst, feature.at(feature_index).x, feature.at(feature_index).y
							, 15, 15, Scalar(255 , 191 , 0));
	}else{
		Draw_Cross(dst, feature.at(feature_index).x, feature.at(feature_index).y
							, 15, 15, Scalar(0 , 255 , 255));
	}//end else
}//end for

inline void GrayProjectionMethod_SelfDefined(const Mat &src , Mat &dst , int &minimal_y , int &minimal_x){
	dst = Frame_wh.clone();

	double row_min_numerical = FLT_MAX;
	double col_min_numerical = FLT_MAX;

	for(int i=0;i<src.rows;++i){
		double sum = 0;
		for(int j=0;j<src.cols;++j){
			sum+=src.at<uchar>(i , j);
		}
		sum/=src.cols;
		if(sum<row_min_numerical){
			row_min_numerical = sum;
			minimal_y = i;
		}
	}

	for(int j=0;j<src.cols;++j){
		double sum = 0;
		for(int i=0;i<src.rows;++i){
			sum+=src.at<uchar>(i , j);
		}
		sum/=src.rows;
		if(sum<col_min_numerical){
			col_min_numerical = sum;
			minimal_x = j;
		}
	}

	line(dst,Point(0 , minimal_y),Point(dst.cols-1 , minimal_y),Scalar(255 , 255 , 100),4,8);
	line(dst,Point(minimal_x , 0),Point(minimal_x , dst.rows-1),Scalar(255 , 255 , 100),4,8);

}

inline void HSVProjectionMethod(const Mat &src , int &maximal_y , int &maximal_x){
		
	Mat colSum;
	double min_col;
	double max_col;
	Point minLoc;
	Point maxLoc;
	reduce(src, colSum, 1, cv::REDUCE_SUM , CV_32FC1);
	minMaxLoc( colSum, &min_col, &max_col, &minLoc, &maxLoc , Mat());
	maximal_y = maxLoc.y;

	Mat rowSum;	
	double min_row;
	double max_row;
	reduce(src, rowSum, 0, cv::REDUCE_SUM , CV_32FC1);
	minMaxLoc( rowSum, &min_row, &max_row, &minLoc, &maxLoc , Mat());
	maximal_x = maxLoc.x;
}

inline void GrayProjectionMethod(const Mat &src,int &minimal_y , int &minimal_x){
	Mat colSum;
	double min_col;
	double max_col;
	Point minLoc;
	Point maxLoc;
	reduce(src, colSum, 0, cv::REDUCE_SUM , CV_32FC1);
	minMaxLoc( colSum, &min_col, &max_col, &minLoc, &maxLoc , Mat());
	minimal_x = minLoc.x;

	Mat rowSum;	
	double min_row;
	double max_row;
	reduce(src, rowSum, 1, cv::REDUCE_SUM , CV_32FC1);
	minMaxLoc( rowSum, &min_row, &max_row, &minLoc, &maxLoc , Mat());
	minimal_y = minLoc.y;


	//line(dst,Point(0 , minimal_y),Point(display_image.cols-1 , minimal_y),Scalar(255 , 255 , 100),4,8);
	//line(dst,Point(minimal_x , 0),Point(minimal_x , display_image.rows-1),Scalar(255 , 255 , 100),4,8);	
}


inline void DisplayInliersOutliersParabola(Mat &Inliers_outliers ,const vector<Point> &inliers	, const vector<Point> &feature 
													, const vector<Point> &outliers , const Mat & Origin_imag_in , const Point &vertexParabola){
	if(inliers.size()==0)return;

	Inliers_outliers = Origin_imag_in.clone();

	for(int i=0;i<outliers.size();++i){
		Draw_Cross(Inliers_outliers, outliers[i].x, outliers[i].y, 15, 15, Scalar(255 , 255 , 100));
	}
	for(int i=0;i<inliers.size();++i){
		Draw_Cross(Inliers_outliers, inliers[i].x, inliers[i].y, 15, 15, Scalar(255 , 255 , 255));
	}	

	if(checkpoint(vertexParabola.x, vertexParabola.y, Inliers_outliers)){
		Draw_Cross(Inliers_outliers, vertexParabola.x, vertexParabola.y, 15, 15, Scalar(255 , 10 , 10));
	}
}
inline void DisplayInliersOutliersEllipseType2(Mat &Inliers_outliers ,const vector<Point> &inlierEllipseFittingPoints
									, const vector<Point> &outlierEllipseFittingPoints , const Mat & Origin_imag_in){
	if(inlierEllipseFittingPoints.size()==0)return;
	Inliers_outliers = Origin_imag_in.clone();

	for(int i=0;i<outlierEllipseFittingPoints.size();++i){		
		Draw_Cross(Inliers_outliers, outlierEllipseFittingPoints[i].x, outlierEllipseFittingPoints[i].y
							, 15, 15, Scalar(0 , 255 , 255));	
	}	
	for(int j=0;j<inlierEllipseFittingPoints.size();++j){
		Draw_Cross(Inliers_outliers, inlierEllipseFittingPoints[j].x, inlierEllipseFittingPoints[j].y
							, 15, 15, Scalar(255 , 191 , 0));			
	}

}

inline void DisplayInliersOutliersEllipse(Mat &Inliers_outliers ,const vector<Point> &inlierEllipseFittingPoints
									, const vector<Point> &feature , const Mat & Origin_imag_in){
	if(inlierEllipseFittingPoints.size()==0)return;
	Inliers_outliers = Origin_imag_in.clone();

	for(int i=0;i<feature.size();++i){
		bool isOutlier = true;
		for(int j=0;j<inlierEllipseFittingPoints.size();++j){
			if(feature[i]==inlierEllipseFittingPoints[j]){
				isOutlier = false;
				Draw_Cross(Inliers_outliers, feature[i].x, feature[i].y
							, 15, 15, Scalar(255 , 191 , 0));
			}
		}
					
		if(isOutlier){
			Draw_Cross(Inliers_outliers, feature[i].x, feature[i].y
							, 15, 15, Scalar(0 , 255 , 255));
		}
	}
}

inline void DisplayInliersOutliersTwoSets(Mat &Inliers_outliers ,const int &return_max_inliers_num1 , const int* inliers_index1
															, const int &return_max_inliers_num2 , const int* inliers_index2
															, const vector<Point> &feature , const Mat &Origin_imag_in)
{
	if(inliers_index1==NULL || inliers_index2==NULL)return;

	Inliers_outliers = Origin_imag_in.clone();
	bool is_inliers = false;
	bool is_inliers1;
	bool is_inliers2;
	for (int i = 0; i < feature.size(); ++i){
		is_inliers1 = false;
		is_inliers2 = false;
		for (int j = 0; j < return_max_inliers_num1; ++j){
			if (i == inliers_index1[j]){
				Draw_Cross(Inliers_outliers, feature[i].x, feature[i].y
							, 15, 15, Scalar(255 , 191 , 0));
				is_inliers1 = true;
				break;
			}
		}
	

		if(!is_inliers1){
			for (int j = 0; j < return_max_inliers_num1; ++j){
				if (i == inliers_index2[j]){
					Draw_Cross(Inliers_outliers, feature[i].x, feature[i].y
								, 15, 15, Scalar(0 , 191 , 0));
					is_inliers2 = true;
					break;
				}
			}
		}

		if (!is_inliers1 && !is_inliers2)
			Draw_Cross(Inliers_outliers, feature , i , false);	
	}


	imshow("Inliers_Outliers" , Inliers_outliers);
}

inline void DisplayFeature(const Mat &image_orig , Mat &dst , const vector<Point> &refineFeaturePoint 
										,const vector<Point> &feature_pts){
	dst = image_orig.clone();
	Mat RefineFeature = image_orig.clone();
	for(int i=0;i<feature_pts.size();++i){
		Draw_Cross(dst, feature_pts[i].x, feature_pts[i].y, 15, 15, Scalar(0,255,100));
	}

	for(int i=0;i<refineFeaturePoint.size();++i){
		Draw_Cross(RefineFeature, refineFeaturePoint[i].x, refineFeaturePoint[i].y, 15, 15, Scalar(0,255,100));
	}
	imshow("AllFeatures" , dst);
	imshow("RefineAllFeatures" , RefineFeature);
}

inline void DisplayFeature(const Mat &image_orig , Mat &dst,const vector<Point> &feature_pts){
	dst = image_orig.clone();	
	for(int i=0;i<feature_pts.size();++i){
		Draw_Cross(dst, feature_pts[i].x, feature_pts[i].y, 15, 15, Scalar(0,255,100));		
	}	
}


inline void DrawEllipse(const Mat &image_orig , Mat &dst  ,const float* ellipse_param 
										, const Point& start_point){

	dst = image_orig.clone();
	float ellipse_axia_a = ellipse_param[0];
	float ellipse_axia_b = ellipse_param[1];
	float cx			  = ellipse_param[2];
	float cy			  = ellipse_param[3];
	float theta		  = ellipse_param[4];


	Point  CircleCenter=cvPoint(cx,cy);
    Size   EllipseAxes=cvSize(ellipse_axia_a,ellipse_axia_b);
    float   RotateAngle=theta*180.0/PI;
    float   StartDrawingAngle=0;
    float   StopDrawingAngle=360;
    Scalar Color=CV_RGB(255,0,0);
    int		 Thickness=7*(image_orig.cols/FRAMEW);
    int		 Shift=0;

    ellipse(dst,CircleCenter,EllipseAxes,RotateAngle,
                 StartDrawingAngle,StopDrawingAngle,Color,Thickness,LINE_AA,Shift);//Draw ellipse


	Draw_Cross(dst, cx, cy, 15, 15, Scalar(0 , 165 , 255));
	//Draw_Cross(dst, start_point.x, start_point.y, 15, 15, Scalar(0 , 0 , 255));	
	//imshow("Ellipse" , dst);
}



inline void DrawPrabolaVer2(Mat &dst  , const float *const parabolaTable , const Mat &OriginalFrame
											, const int &interDist , const bool &getEyelidRegion){
	if(getEyelidRegion){
		dst = OriginalFrame.clone();
		vector<Point> pt;
		 for (int x = 0; x < FRAMEW-1; x+=interDist){         
			if(x+interDist>FRAMEW-1)continue;
			line(dst, Point(x,int(parabolaTable[x])), Point(x+interDist,int(parabolaTable[x+interDist])), Scalar(200 , 225 , 100) , 20 , LINE_AA);
		 }
	}
}

inline void DrawPrabolaVer3(Mat &dst , const float* const parabolaTable, const int &interDist
											, const bool &getEyelidRegion){	
	if(getEyelidRegion){
		vector<Point> pt;
		 for (int x = 0; x < FRAMEW-1; x+=interDist){
			 if(x+interDist>FRAMEW-1)continue;
			 line(dst, Point(x,int(parabolaTable[x])), Point(x+interDist,int(parabolaTable[x+interDist])), Scalar(200 , 225 , 100) , 20 , LINE_AA);
		 }     
	}
}

inline void DrawCubic(Mat &dst  , const float *const cubic_para , const Mat &OriginalFrame){
	dst = OriginalFrame.clone();
	vector<Point> pt;
	 for (int x = 0; x < FRAMEW; ++x)
     {
         float y = cubic_para[0] * double(x) * double(x)*double(x) + cubic_para[1] * double(x) * double(x)+ cubic_para[2]*double(x)
						+cubic_para[3];
         pt.push_back(Point(x, int(y)));
     }     
     for (int i = 0; i < pt.size()-1; ++i)
     {
         line(dst, pt[i], pt[i+1], Scalar(200 , 225 , 100) , 20 , LINE_AA );
     }     
}


inline void DrawPrabola(Mat &dst  ,Mat &bool_testing_equation_paint , float *parabola_para , const Mat &OriginalFrame){
	dst = OriginalFrame.clone();
	enum  {
		Type_full = 255,
		Type_empty = 0,
		Type_Intersection = 128
	};

	int height = dst.rows;
	int width = dst.cols;
	bool_testing_equation_paint = Mat::zeros(dst.rows , dst.cols , CV_8UC1);	

	const double rad_to_theta_cons = M_PI/180.f;
	int ksize = 9;
	float confident_ratio = 0.9;
	float max_boundary_ROI;
	
	
	for(int i=0;i<dst.rows;++i){
		for(int j=0;j<dst.cols;++j){
			if(!checkpoint(j ,i , bool_testing_equation_paint))
				continue;
			
			/*if(parabola_para[0]*powf(j , 2.f)+parabola_para[1]*j*(height-i)+parabola_para[2]*powf((height-i) , 2.f)+
				parabola_para[3]*j+parabola_para[4]*(height-i)+parabola_para[5]<0)
					bool_testing_equation_paint.at<uchar>(i , j) = Type_full;*/
					//testing_equation_paint.at<uchar>(i , j) = 255;

			if(parabola_para[0]*powf(j , 2.f)+parabola_para[1]*j+parabola_para[2]-(height-i)<0)
					bool_testing_equation_paint.at<uchar>(i , j) = Type_full;
					//testing_equation_paint.at<uchar>(i , j) = 255;
		}
	}	


	

	for(int i=0;i<dst.rows;++i){
		for(int j=0;j<dst.cols;++j){
			if(!checkpoint(j ,i , dst))
				continue;

			if(checkpoint(j-1 ,i , bool_testing_equation_paint) && bool_testing_equation_paint.at<uchar>(i , j-1)==0 && bool_testing_equation_paint.at<uchar>(i , j)==255){				
				circle(dst,Point(j , i), 2, Scalar(255 , 150 , 0), 2);
				
				/*dst(i , j)[0] = 255;
				dst(i , j)[1] = 0;
				dst(i , j)[2] = 0;*/
			}else if(checkpoint(j+1 ,i , bool_testing_equation_paint) && bool_testing_equation_paint.at<uchar>(i , j+1)==0 && bool_testing_equation_paint.at<uchar>(i , j)==255){
				circle(dst,Point(j , i), 2, Scalar(255 , 150 , 0), 2);

				/*dst(i , j)[0] = 255;
				dst(i , j)[1] = 0;
				dst(i , j)[2] = 0;				*/	
			}else if(checkpoint(j ,i-1 , bool_testing_equation_paint) && bool_testing_equation_paint.at<uchar>(i-1 , j)==0 && bool_testing_equation_paint.at<uchar>(i , j)==255){
				circle(dst,Point(j , i), 2, Scalar(255 , 150 , 0), 2);
				
				/*dst(i , j)[0] = 255;
				dst(i , j)[1] = 0;
				dst(i , j)[2] = 0;			*/		
			}else if(checkpoint(j ,i+1 , bool_testing_equation_paint) && bool_testing_equation_paint.at<uchar>(i+1 , j)==0 && bool_testing_equation_paint.at<uchar>(i , j)==255){
				circle(dst,Point(j , i), 2, Scalar(255 , 150 , 0), 2);
				
				/*dst(i , j)[0] = 255;
				dst(i , j)[1] = 0;
				dst(i , j)[2] = 0;		*/					
			}
		}
	}		
}

inline void CenterCalculatUsingMoment(const vector<Point> &max_contours , int &posX , int &posY){
	Moments oMoments = moments(max_contours);

	double dM01 = oMoments.m01;
	double dM10 = oMoments.m10;
	double dArea = oMoments.m00;
	posX = dM10 / dArea;
	posY = dM01 / dArea;       

	//std::cout << ">>Opencv dM10 = " << dM10 << std::endl;
	//std::cout << ">>Opencv dM01 = " << dM01 << std::endl;
	//std::cout << ">>Opencv dM00 = " << dArea << std::endl;

}

inline void CenterCalculatUsingMoment(const Mat &Src_contours , int &posX , int &posY){
	Moments oMoments = moments(Src_contours , true);

	double dM01 = oMoments.m01;
	double dM10 = oMoments.m10;
	double dArea = oMoments.m00;
	posX = dM10 / dArea;
	posY = dM01 / dArea;

	//std::cout << ">>Opencv dM10 = " << dM10 << std::endl;
	//std::cout << ">>Opencv dM01 = " << dM01 << std::endl;
	//std::cout << ">>Opencv dM00 = " << dArea << std::endl;
}

inline void CenterCalculatUsingMomentSerial(const Mat &Src_contours, int &posX, int &posY) {
	double dM01 = 0;
	double dM10 = 0;
	double dM00 = 0;

	for (int r = 0; r<Src_contours.rows; ++r) {
		for (int c = 0; c<Src_contours.cols; ++c) {
			dM01 += (Src_contours.at<uchar>(r, c))*(r+1);
		}
	}
	for (int r = 0; r<Src_contours.rows; ++r) {
		for (int c = 0; c<Src_contours.cols; ++c) {
			dM10 += (Src_contours.at<uchar>(r, c))*(c + 1);
		}
	}
	for (int r = 0; r<Src_contours.rows; ++r) {
		for (int c = 0; c<Src_contours.cols; ++c) {
			dM00 += (Src_contours.at<uchar>(r, c));
		}
	}

	posX = dM10 / dM00;
	posY = dM01 / dM00;
}

inline void CenterCalculatUsingMomentParallel(const Mat &Src_contours, int &posX, int &posY, const int thread_num) {
	double *m10_array = new double [thread_num]();
	double *m01_array = new double [thread_num]();
	double *m00_array = new double [thread_num]();
	double dM10 = 0;
	double dM01 = 0;
	double dM00 = 0;
	int *start_loc = new int[thread_num]();
	int *end_loc = new int[thread_num]();
	int avg_rows = Src_contours.rows / thread_num;
	int extra_rows = Src_contours.rows % thread_num;

	for (int i = 0; i<thread_num; ++i) {
		int my_rank_rows = (i<(extra_rows)) ? (avg_rows + 1) : avg_rows;
		start_loc[i] = (i == 0) ? 0 : end_loc[i - 1] + 1;
		end_loc[i] = start_loc[i] + (my_rank_rows - 1);
	}

	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_moment(Src_contours, m10_array, m01_array, m00_array, start_loc, end_loc, thread_num));//this can handle the case that image.rows is not divisible by thread_num
	//cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_moment(Src_contours, m10_array, m01_array, m00_array, thread_num));

	for (int i = 0; i < thread_num; ++i) {
		dM10 += m10_array[i];
		dM01 += m01_array[i];
		dM00 += m00_array[i];
	}

	//std::cout << ">>Parallel dM10 = " << dM10 << std::endl;
	//std::cout << ">>Parallel dM01 = " << dM01 << std::endl;
	//std::cout << ">>Parallel dM00 = " << dM00 << std::endl;
	posX = dM10 / dM00;
	posY = dM01 / dM00;

	delete [] m10_array;
	delete [] m01_array;
	delete [] m00_array;
}

inline void CenterCalculatUsingMomentParallel(const vector<Point> &Src_contours, int &posX, int &posY, const int thread_num) {
	double *m10_array = new double[thread_num]();
	double *m01_array = new double[thread_num]();
	double *m00_array = new double[thread_num]();
	double dM10 = 0;
	double dM01 = 0;
	double dM00 = 0;
	int *start_loc = new int[thread_num]();
	int *end_loc = new int[thread_num]();
	int avg_rows = Src_contours.size() / thread_num;
	int extra_rows = Src_contours.size() % thread_num;

	for (int i = 0; i<thread_num; ++i) {
		int my_rank_rows = (i<(extra_rows)) ? (avg_rows + 1) : avg_rows;
		start_loc[i] = (i == 0) ? 0 : end_loc[i - 1] + 1;
		end_loc[i] = start_loc[i] + (my_rank_rows - 1);
	}

	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_moment_vector(Src_contours, m10_array, m01_array, m00_array, start_loc, end_loc, Src_contours.size(), thread_num));

	for (int i = 0; i < thread_num; ++i) {
		dM10 += m10_array[i];
		dM01 += m01_array[i];
		dM00 += m00_array[i];
	}

	if (fabs(dM00) > FLT_EPSILON)
	{
		double db1_2, db1_6;

		if (dM00 > 0)
		{
			db1_2 = 0.5;
			db1_6 = 0.16666666666666666666666666666667;
		}
		else
		{
			db1_2 = -0.5;
			db1_6 = -0.16666666666666666666666666666667;
		}

		// spatial moments
		dM00 = dM00 * db1_2;
		dM10 = dM10 * db1_6;
		dM01 = dM01 * db1_6;
	}

	//std::cout << ">>Parallel dM10 = " << dM10 << std::endl;
	//std::cout << ">>Parallel dM01 = " << dM01 << std::endl;
	//std::cout << ">>Parallel dM00 = " << dM00 << std::endl;

	posX = dM10 / dM00;
	posY = dM01 / dM00;

	delete [] m10_array;
	delete [] m01_array;
	delete [] m00_array;
	delete [] start_loc;
	delete [] end_loc;
}

inline bool MinimalIrisColorProcess(const Mat &Src , Point &eyeCoarseCenter , const int &size_gaussian 
													, Mat &IrisContour , vector<Point> &IrisContoursPoints , bool &getIrisContourPoints
													, float &irisContour_size)
{	
	Mat GrayGaussian = Mat::zeros(Src.size(), CV_8UC1);
	Mat GrayOtsu = Mat::zeros(Src.size(), CV_8UC1);				

	//GaussianBlur( Src, GrayGaussian, Size(size_gaussian,size_gaussian) , 0);
	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_gau(Src, GrayGaussian, size_gaussian, thread_num));
	//threshold(GrayGaussian, GrayOtsu, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	ParallelOtsu(GrayGaussian, GrayOtsu, cv::THRESH_BINARY, thread_num);

	double time_start1 = getTickCount();
	if(!FindMAXConnextedComponent(GrayOtsu , IrisContoursPoints , IrisContour)){
		getIrisContourPoints = false;
		return false;
	}

	irisContour_size = countNonZero(IrisContour);
	time_eye_position_detection_coarse_iris_location_color_serial.push_back(getTickCount() - time_start1);
	//CenterCalculatUsingMoment(IrisContoursPoints , eyeCoarseCenter.x , eyeCoarseCenter.y);
	//std::cout << "OpenCV memoent = (" << eyeCoarseCenter.x << ", " << eyeCoarseCenter.y << ")" << std::endl;

	CenterCalculatUsingMomentParallel(IrisContoursPoints, eyeCoarseCenter.x, eyeCoarseCenter.y, thread_num);
	//std::cout << "CenterCalculatUsingMomentParallel memoent = (" << eyeCoarseCenter.x << ", " << eyeCoarseCenter.y << ")" << std::endl;

	//waitKey(0);

	getIrisContourPoints = true;
	return true;	
}

inline void  IntegralImage(const Mat &src , Mat &dst){
	dst = Mat::zeros(src.rows , src.cols , CV_32FC1);
	for (int i= 0; i < src.rows; ++i){	
		float sum = 0;
		for (int j = 0; j < src.cols; ++j){
			sum += src.at<float>(i , j);//image[j, i]


			int check_i = i-1;

			if(check_i<0){
				dst.at<float>(i , j) = sum;		
			}else{
				dst.at<float>(i , j) = dst.at<float>(check_i , j)+sum;			
			}
		}
	}
}

inline bool RefineEyeCornerROI( const int &roi_LeftWidth , const int &roi_LeftHeight , const int &roi_RightWidth , const int &roi_RightHeight 
												 , Point &rectLeftCornerStartPoint , Point &rectRightCornerStartPoint
												 , const Point &leftEyeCornerUserInput , const Point &rightEyeCornerUserInput
												 , int &leftCorrect_x_left , int &leftCorrect_x_right , int &leftCorrect_y_up , int & leftCorrect_y_down
												 , int &rightCorrect_x_left , int &rightCorrect_x_right , int &rightCorrect_y_up , int & rightCorrect_y_down
												 , int &width_LeftROIforFollowingTM ,int &height_LeftROIforFollowingTM
												 , int &width_RightROIforFollowingTM ,int &height_RightROIforFollowingTM)
{
	int half_roi_LeftWidth = roi_LeftWidth/2;
	int half_roi_LeftHeight = roi_LeftHeight/2;
	int half_roi_RightWidth = roi_RightWidth/2;
	int half_roi_RightHeight = roi_RightHeight/2;

	//---------------------------------Determine rectLeftCornerStartPoint && leftCorrect_x_left && leftCorrect_x_right--------------------------------------//
	if(leftEyeCornerUserInput.x - half_roi_LeftWidth<0){
		rectLeftCornerStartPoint.x = 0;
		leftCorrect_x_left = leftEyeCornerUserInput.x;
	}else if(leftEyeCornerUserInput.x - half_roi_LeftWidth>=FRAMEW){
		leftCorrect_x_left = 0;
		leftCorrect_x_right = 0;
		printf("Left eye corner.x1 is not in the image range\n");return false;
	}else{
		rectLeftCornerStartPoint.x = leftEyeCornerUserInput.x - half_roi_LeftWidth;
		leftCorrect_x_left = half_roi_LeftWidth;
	}

	if(leftEyeCornerUserInput.x + half_roi_LeftWidth>=FRAMEW){		
		leftCorrect_x_right = FRAMEW -1 - leftEyeCornerUserInput.x;
	}else if(leftEyeCornerUserInput.x + half_roi_LeftWidth<0){
		leftCorrect_x_right = 0;
		leftCorrect_x_left = 0;
		printf("Left eye corner.x2 is not in the image range\n");return false;
	}else{
		leftCorrect_x_right = half_roi_LeftWidth;
	}

	if(leftEyeCornerUserInput.y - half_roi_LeftHeight<0){
		rectLeftCornerStartPoint.y = 0;
		leftCorrect_y_up = leftEyeCornerUserInput.y;
	}else if(leftEyeCornerUserInput.y - half_roi_LeftHeight>=FRAMEH){
		leftCorrect_y_up = 0;
		leftCorrect_y_down = 0;
		printf("Left eye corner.y is not in the image range\n");return false;
	}else{
		rectLeftCornerStartPoint.y = leftEyeCornerUserInput.y - half_roi_LeftHeight;
		leftCorrect_y_up = half_roi_LeftHeight;
	}

	if(leftEyeCornerUserInput.y + half_roi_LeftHeight>=FRAMEH){		
		leftCorrect_y_down = FRAMEH -1 - leftEyeCornerUserInput.y;
	}else if(leftEyeCornerUserInput.y + half_roi_LeftHeight< 0){
		leftCorrect_y_up = 0;
		leftCorrect_y_down = 0;
		printf("Left eye corner.y is not in the image range\n");return false;
	}else{
		rectLeftCornerStartPoint.y = leftEyeCornerUserInput.y - half_roi_LeftHeight;
		leftCorrect_y_down = half_roi_LeftHeight;
	}

	//---------------------------------Determine width_LeftROIforFollowingTM && height_LeftROIforFollowingTM------------------------------------------//
	width_LeftROIforFollowingTM = leftCorrect_x_left+leftCorrect_x_right;
	height_LeftROIforFollowingTM = leftCorrect_y_up+leftCorrect_y_down;


	//---------------------------------Determine rectRightCornerStartPoint && rightCorrect_x_left && rightCorrect_x_right--------------------------------------//
	if(rightEyeCornerUserInput.x - half_roi_RightWidth<0){
		rectRightCornerStartPoint.x = 0;
		rightCorrect_x_left = leftEyeCornerUserInput.x;
	}else if(rightEyeCornerUserInput.x - half_roi_RightWidth>=FRAMEW){
		rightCorrect_x_left = 0;
		rightCorrect_x_right = 0;
		printf("Right eye corner.x is not in the image range\n");return false;
	}else{
		rectRightCornerStartPoint.x = rightEyeCornerUserInput.x - half_roi_RightWidth;
		rightCorrect_x_left = half_roi_RightWidth;
	}

	if(rightEyeCornerUserInput.x + half_roi_RightWidth>=FRAMEW){		
		rightCorrect_x_right = FRAMEW -1 - rightEyeCornerUserInput.x;
	}else if(rightEyeCornerUserInput.x + half_roi_RightWidth<0){
		rightCorrect_x_left = 0;
		rightCorrect_x_right = 0;
		printf("Right eye corner.x is not in the image range\n");return false;
	}else{
		rightCorrect_x_right = half_roi_RightWidth;
	}

	if(rightEyeCornerUserInput.y - half_roi_RightHeight<0){
		rectRightCornerStartPoint.y = 0;
		rightCorrect_y_up = rightEyeCornerUserInput.y;
	}else if(rightEyeCornerUserInput.y - half_roi_RightHeight>=FRAMEH){
		rightCorrect_y_up = 0;
		rightCorrect_y_down = 0;
		printf("Right eye corner.y is not in the image range\n");return false;
	}else{
		rectRightCornerStartPoint.y = rightEyeCornerUserInput.y - half_roi_RightHeight;
		rightCorrect_y_up = half_roi_RightHeight;
	}

	if(rightEyeCornerUserInput.y + half_roi_RightHeight>=FRAMEH){		
		rightCorrect_y_down = FRAMEH -1 - rightEyeCornerUserInput.y;
	}else if(rightEyeCornerUserInput.y + half_roi_LeftHeight< 0){
		rightCorrect_y_up = 0;
		rightCorrect_y_down = 0;
		printf("Right eye corner.y is not in the image range\n");return false;
	}else{
		rectRightCornerStartPoint.y = rightEyeCornerUserInput.y - half_roi_RightHeight;
		rightCorrect_y_down = half_roi_RightHeight;
	}

	//---------------------------------Determine width_RightROIforFollowingTM && height_RightROIforFollowingTM------------------------------------------//
	width_RightROIforFollowingTM = rightCorrect_x_left+rightCorrect_x_right;
	height_RightROIforFollowingTM = rightCorrect_y_up+rightCorrect_y_down;

	return true;
}

inline void minmax(const Mat &result , Point &matchLoc )
{
  double minVal, maxVal;
  Point  minLoc, maxLoc;

  minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
  matchLoc = maxLoc;  
}

//inline void ROIEyeCornerTrackingKLT(const Mat &ROI_Leftcorner , const Mat &ROI_Rightcorner , const Point &rectLeftCornerStartPoint){
//	int ksize = 41;
//	Mat ROI_LeftMedian;
//	Mat ROI_RightMedian;
//	Mat ROI_LeftFilteredResult = ROI_Leftcorner.clone();
//	Mat ROI_RightFilteredResult;
//
//	vector<uchar> status;
//	vector<float> err;
//	if(PrevGray.empty())
//        ROI_Leftcorner.copyTo(PrevGray);
//
//	if(PrevGray.size()!=ROI_Leftcorner.size()){
//		Rect myROI(0, 0, ROI_Leftcorner.cols, ROI_Leftcorner.rows);
//		PrevGray = ROI_Leftcorner(myROI);
//	}	
//
//	cout<<endl<<"in:"<<endl;
//	cout<<"points[0][0] = "<<points[0][0]<<endl;
//
//	cout<<"PrevGray.size() = "<<PrevGray.size()<<endl;
//	cout<<"ROI_Leftcorner.size() = "<<ROI_Leftcorner.size()<<endl;
//	
//	
//	calcOpticalFlowPyrLK(PrevGray, ROI_Leftcorner, points[0], points[1], status, err, winSize,
//                                 3, termcrit, 0, 0.001);
//
//	if(norm((Point2f(22,313) - (Point2f)rectLeftCornerStartPoint - points[1][0]))>50){
//		points[1][0] = (Point2f(22,313) - (Point2f)rectLeftCornerStartPoint);
//	}
//
//	cout<<endl<<"in:"<<endl;
//	cout<<"points[0][0] = "<<points[0][0]<<endl;
//	cout<<"points[1][0] = "<<points[1][0]<<endl;
//	cout<<"status.size = "<<status.size()<<endl;
//	cout<<"status[0] = "<<(int)status.at(0)<<endl;
//
//	imshow("PrevGray" , PrevGray);
//	imshow("ROI_Leftcorner" , ROI_Leftcorner);
//
//
//
//
//	//medianBlur(ROI_Leftcorner , ROI_LeftMedian , ksize);		
//	//medianBlur(ROI_Rightcorner , ROI_RightMedian , ksize);	
//
//
//	//float filterKernelLeftEye[4][6] = {
//	//	{-1 , -1 , -1 , 1 , 1 , 1},
//	//	{-1 , -1 , 1 , 1 , 1 , 1},
//	//	{-1 , 1 , 1 , 1 , 1 , 1},
//	//	{-1, -1, -1 , -1, -1,-1}
//	//};
//
//
//	////float filterKernelLeftEye[2][3] = {
//	////	{-1 , 1, 1},
//	////	{-1 , -1 , -1}
//	////};
//
//	////float filterKernelLeftEye[8][14] = {
//	////	{-1 , -1 , -1 , -1 , -1 , -1 ,-1 ,1 ,1 ,1 , 1 , 1 , 1 , 1},
//	////	{-1 , -1 , -1 , -1 , -1 , -1 ,1 ,1 ,1 ,1 , 1 , 1 , 1 , 1},
//	////	{-1 , -1 , -1 , -1 , -1 , 1 ,1 ,1 ,1 ,1 , 1 , 1 , 1 , 1},
//	////	{-1 , -1 , -1 , -1 , 1 , 1 ,1 ,1 ,1 ,1 , 1 , 1 , 1 , 1},
//	////	{-1 , -1 , -1 , 1 , 1 , 1 ,1 ,1 ,1 ,1 , 1 , 1 , 1 , 1},
//	////	{-1 , -1 , 1 , 1 , 1 , 1 ,1 ,1 ,1 ,1 , 1 , 1 , 1 , 1},
//	////	{-1 , 1 , 1 , 1 , 1 , 1 ,1 ,1 ,1 ,1 , 1 , 1 , 1 , 1},
//	////	{-1 , -1 , -1 , -1 , -1 , -1 ,-1 ,-1 ,-1 ,-1 , -1 , -1 , -1 , -1},
//	////};
//
//	//float filterKernelRightEye[4][6] = {
//	//	{1 , 1 , 1 , -1 , -1 , -1},
//	//	{1 , 1 , 1 , 1 , -1 , -1},
//	//	{1 , 1 , 1 , 1 , 1 , -1},
//	//	{-1, -1, -1 , -1, -1,-1}
//	//};
//
//
//	//Mat filterLeftEye(4 , 6, CV_32FC1 , filterKernelLeftEye);
//	//Mat filterRightEye(4 , 6 , CV_32FC1 , filterKernelRightEye);
//	//	
//	//Point anchor = Point( -1, -1 );
//	//int delta = 0;
//	//int ddepth = -1;
//
//	//cv::filter2D(ROI_LeftMedian, ROI_LeftFilteredResult, ddepth , filterLeftEye, anchor, delta, BORDER_DEFAULT );
//	//cv::filter2D(ROI_RightMedian, ROI_RightFilteredResult, ddepth , filterRightEye, anchor, delta, BORDER_DEFAULT );	
//
//	//double minVal,maxVal;
//	//Point minLoc , maxLoc;
//
//	//minMaxLoc(ROI_LeftFilteredResult, &minVal, &maxVal, &minLoc, &maxLoc);
//	
//	Draw_Cross(ROI_LeftFilteredResult, points[1][0].x, points[1][0].y, 10, 10, Scalar(255 , 255 , 255));
//	cv::imshow("ROI_LeftFilteredResult" ,ROI_LeftFilteredResult);
//	//cv::imshow("ROI_RightFilteredResult" ,ROI_RightFilteredResult);
//}



//inline void KLT(){
//	vector<Point2f> tmp;
//	tmp.push_back(leftEyeCorner);
//	cornerSubPix( Frame_Gray, tmp, winSize, cvSize(-1,-1), termcrit);
//
//	if(frame_number==1){		
//		points[0].push_back(tmp[0]);			
//	}else{
//		points[0][0] = tmp[0];
//	}
//	
//
//	cout<<"out:"<<endl;
//	cout<<"points[0][0] = "<<points[0][0]<<endl;
//	cout<<"rectLeftCornerStartPoint = "<<rectLeftCornerStartPoint<<endl;
//
//	points[0][0] = points[0][0] -  (Point2f)rectLeftCornerStartPoint;	
//	ROIEyeCornerTrackingKLT(ROI_Leftcorner , ROI_Rightcorner , rectLeftCornerStartPoint);	
//	points[1][0] = points[1][0] +  (Point2f)rectLeftCornerStartPoint;	
//
//	cout<<"out:"<<endl;
//	cout<<"points[1][0] = "<<points[1][0]<<endl;
//	cout<<"rectLeftCornerStartPoint = "<<rectLeftCornerStartPoint<<endl;
//	
//	
//	std::swap(points[1], points[0]);
//    cv::swap(PrevGray, ROI_Leftcorner);	
//	points[1].clear();
//
//}

inline Point2f DerivativeParabola(const float * const conic_par , const Point2f &pt){
	return Point2f(2*conic_par[0]*pt.x+conic_par[1]
						,-1);
}

inline float ErrorParabolaEOF2(const float *const conic_par , const Point2f &pt){
	float dis_error = conic_par[0]*pt.x*pt.x + 
	         				conic_par[1]*pt.x+
	         				conic_par[2]-(pt.y);
	Point2f gradParabola = DerivativeParabola(conic_par, pt);
	float magnitude = 	norm(gradParabola - Point2f(0 , 0));

	dis_error/=magnitude;
	return dis_error;
}

inline void RefineUpperEyelidFeaturePts(vector<Point> &originalFtPts , vector<Point> &newFtPts 
															, const float * const lowerParabola_param , const float &dis_thresholdSoft_lowerEyelid
															, const vector<Point> &lower_parabola_outlier , const Point &vertexParabolaLower)
{//Use the current frame detected parabola
	for(int i=0;i<originalFtPts.size();++i){		
		if(ErrorParabolaEOF2(lowerParabola_param , originalFtPts[i])>dis_thresholdSoft_lowerEyelid){
			newFtPts.push_back(originalFtPts[i]);
		}
	}

	for(int i=0;i<lower_parabola_outlier.size();++i){
		if(lower_parabola_outlier[i].y>vertexParabolaLower.y)continue;
		newFtPts.push_back(lower_parabola_outlier[i]);
	}
}

inline void RefineEyelidFeaturePts(const vector<Point> &originalUpperFtPts , vector<Point> &newUpperFtPts 
													, const vector<Point> &originalLowerFtPts , vector<Point> &newLowerFtPts
													, const float * const lowerParabola_paramLast , const float &dis_thresholdSoft_lowerEyelidLast
													, const float * const upperParabola_paramLast , const float &dis_thresholdSoft_upperEyelidLast
													, const Point &vertexParabolaLowerLast	, const Point &vertexParabolaUpperLast
													, const bool &bigmotionEllipseCenter)
{//Use last frame detected parabola	
	//Upper
	if((lowerParabola_paramLast[0]==0 && lowerParabola_paramLast[1]==0 && lowerParabola_paramLast[2]==0)
		|| (vertexParabolaLowerLast.y<vertexParabolaUpperLast.y)){
		newUpperFtPts = originalUpperFtPts;		
	}else{
		for(int i=0;i<originalUpperFtPts.size();++i){
			float error_paraLow_FtUp = fabs(ErrorParabolaEOF2(lowerParabola_paramLast , originalUpperFtPts[i]));
			if(error_paraLow_FtUp>dis_thresholdSoft_lowerEyelidLast){
				if(!bigmotionEllipseCenter){
					float error_paraUp_FtUp = fabs(ErrorParabolaEOF2(upperParabola_paramLast , originalUpperFtPts[i]));
					if(error_paraUp_FtUp<dis_thresholdSoft_upperEyelidLast){
						newUpperFtPts.push_back(originalUpperFtPts[i]);
					}
				}else{
					if(originalUpperFtPts[i].y<vertexParabolaLowerLast.y){
						newUpperFtPts.push_back(originalUpperFtPts[i]);
					}
				}
			}else{
				if(!bigmotionEllipseCenter){
					float error_paraUp_FtUp = fabs(ErrorParabolaEOF2(upperParabola_paramLast , originalUpperFtPts[i]));
					if(error_paraUp_FtUp>dis_thresholdSoft_upperEyelidLast){
						if(originalUpperFtPts[i].y>vertexParabolaUpperLast.y)
							newLowerFtPts.push_back(originalUpperFtPts[i]);
					}else{
						newUpperFtPts.push_back(originalUpperFtPts[i]);
					}
				}
			}
		}
	}


	//Lower
	if(upperParabola_paramLast[0]==0 && upperParabola_paramLast[1]==0 && upperParabola_paramLast[2]==0
		|| (vertexParabolaLowerLast.y<vertexParabolaUpperLast.y)){	
		newLowerFtPts = originalLowerFtPts;		
	}else{
		for(int i=0;i<originalLowerFtPts.size();++i){
			float error_paraUp_FtLow = fabs(ErrorParabolaEOF2(upperParabola_paramLast , originalLowerFtPts[i]));		
			if(error_paraUp_FtLow>dis_thresholdSoft_upperEyelidLast){	
				if(!bigmotionEllipseCenter){
					float error_paraLow_FtLow = fabs(ErrorParabolaEOF2(lowerParabola_paramLast , originalLowerFtPts[i]));
					if(error_paraLow_FtLow<dis_thresholdSoft_lowerEyelidLast){
						newLowerFtPts.push_back(originalLowerFtPts[i]);
					}
				}else{
					if(originalLowerFtPts[i].y>vertexParabolaUpperLast.y){
						newLowerFtPts.push_back(originalLowerFtPts[i]);
					}	
				}
			}else{
				if(!bigmotionEllipseCenter){
					float error_paraLow_FtLow = fabs(ErrorParabolaEOF2(lowerParabola_paramLast , originalLowerFtPts[i]));
					if(error_paraLow_FtLow>dis_thresholdSoft_lowerEyelidLast){
						if(originalLowerFtPts[i].y<vertexParabolaLowerLast.y)
							newUpperFtPts.push_back(originalLowerFtPts[i]);
					}else{
						newLowerFtPts.push_back(originalLowerFtPts[i]);
					}
				}
			}
		}
	}
}

inline bool EyelidMaskGeneration(const float * const lowerParabola_param , const float * const upperParabola_param
													 ,	Mat &Dst , const Point &vertexUpper , const Point &vertexLower 
													 , float *&lowerParabolaTable , float *&upperParabolaTable)
{
	if(( lowerParabola_param[0]==0 && lowerParabola_param[1]==0 && lowerParabola_param[2]==0 )
		|| ( upperParabola_param[0]==0 && upperParabola_param[1]==0 && upperParabola_param[2]==0)
		|| (vertexUpper.y>vertexLower.y))return false;
	
	Dst = Mat::zeros(FRAMEH , FRAMEW , CV_8UC1);

	lowerParabolaTable = new float [FRAMEW]();
	upperParabolaTable = new float [FRAMEW]();

	for(int x = 0; x < FRAMEW; ++x){
         lowerParabolaTable[x] = lowerParabola_param[0] * double(x) * double(x) 
											+ lowerParabola_param[1] * double(x) + lowerParabola_param[2];    
		 upperParabolaTable[x] = upperParabola_param[0] * double(x) * double(x) 
											+ upperParabola_param[1] * double(x) + upperParabola_param[2];         
     }    
		
	for(int j=0;j<Dst.cols;++j){
		for(int i=0;i<Dst.rows;++i){
			if(i>upperParabolaTable[j] && i < lowerParabolaTable[j]){
				Dst.at<uchar>(i , j) = 255;
			}
		}
	}  
	return true;
}

inline void ScleraMaskPreprocessing(const Mat &SrcMask , Mat &DstMask , const int &size_gaussian){
	DstMask = Mat::zeros(SrcMask.rows , SrcMask.cols , CV_8UC1);	
	Mat SrcMask_dilation;
	Mat SrcMask_gaussian;
	Mat SrcMask_med;	
	int morph_size = 5*SrcMask.rows/(float)FRAMEH;

	GaussianBlur( SrcMask, SrcMask_gaussian, Size(size_gaussian,size_gaussian) , 0);	

	threshold(SrcMask_gaussian, SrcMask_gaussian, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	
	medianBlur(SrcMask_gaussian, SrcMask_med, 21);
	
	Mat tmp = SrcMask_med.clone();
	for(int i=0;i<2;++i){		
		Morphology_Operations(tmp, SrcMask_dilation , MORPH_DILATE, morph_size,  MORPH_CROSS);
		tmp = SrcMask_dilation.clone();
	}


	DstMask = SrcMask_dilation.clone();
	

	//imshow("DstMask" , DstMask);
	//imshow("SrcMask_gaussian" , SrcMask_gaussian);
	//imshow("SrcMask_dilation" , SrcMask_dilation);
	//imshow("SrcMask_med" , SrcMask_med);
	//imshow("SrcMask" , SrcMask);
}

inline void X_DirectedGradientGeneration(const Mat &Gray_openingGaussian , Mat &Grad_X_Thresh_Pop){	
	Mat Grad_X;		
	Mat ABS_Grad_X = Mat::zeros(Gray_openingGaussian.size()  ,CV_8UC1);
	Mat ScharrX_tmp = Mat::zeros(Gray_openingGaussian.size()  ,CV_16S);
	Mat ConvertScaleAbsX_tmp = Mat::zeros(Gray_openingGaussian.size()  ,CV_8UC1);
	Mat ABS_Grad_XHisto;
	int *hist = new int[256]();
	//int *cu_hist = new int[256]();
	std::mutex mtx;
	float goal_percentGradthreshold = 0.1*FRAMEW*FRAMEH;
	int thre_populationGray;
	int sum_populationGray = 0;
	double *min_val_cand = new double[thread_num];
	double *max_val_cand = new double[thread_num];
	double min_val = FLT_MAX;
	double max_val = -FLT_MAX;

	//X Directed Gradient
	//Scharr( Gray_openingGaussian, Grad_X, CV_16S, 1, 0, 3, 0, BORDER_DEFAULT );
	//convertScaleAbs( Grad_X, ABS_Grad_X );
	//normalize(ABS_Grad_X, ABS_Grad_X, 0, 255, NORM_MINMAX, CV_8UC1);//L
	
	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_scharrX(Gray_openingGaussian, ScharrX_tmp, thread_num));
	//cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_convertScaleAbs(ScharrX_tmp, ConvertScaleAbsX_tmp, thread_num));
	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_convertScaleAbs_findMinMax(ScharrX_tmp, ConvertScaleAbsX_tmp, min_val_cand, max_val_cand, thread_num));
	for (int i = 0; i < thread_num; ++i) {
		if (min_val > min_val_cand[i]) {
			min_val = min_val_cand[i];
		}
		if (max_val < max_val_cand[i]) {
			max_val = max_val_cand[i];
		}
	}


	//cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_normalize(ConvertScaleAbsX_tmp, ABS_Grad_X, thread_num));
	//normalize(ConvertScaleAbsX_tmp, ABS_Grad_X, 0, 255, NORM_MINMAX, CV_8UC1);//L
	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_converto_min_max(ConvertScaleAbsX_tmp, ABS_Grad_X, min_val, max_val, 0, 255, thread_num));

	
	
	//Histogram Grad_X
	//CalcHistogram(ABS_Grad_X , ABS_Grad_XHisto);	
	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_hist_pure(ABS_Grad_X, hist, mtx, thread_num));
	//cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_hist_and_cumulative_pure(ABS_Grad_X, cu_hist, mtx, thread_num));


	//for(int i=ABS_Grad_XHisto.rows-1;i>0;--i){
	//	sum_populationGray+=ABS_Grad_XHisto.at<float>(0 , i);
	//	if(sum_populationGray>goal_percentGradthreshold){
	//		thre_populationGray = i;
	//		break;
	//	}
	//}

	for(int i=255;i>0;--i){
		sum_populationGray+= hist[i];
		if(sum_populationGray>goal_percentGradthreshold){
			thre_populationGray = i;
			break;
		}
	}

	//for(int i=0;i<256;++i){
	//	if(cu_hist[i]>0.9*FRAMEH*FRAMEH){
	//		thre_populationGray = i-1;
	//		break;
	//	}
	//}



	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_threBinary(ABS_Grad_X, Grad_X_Thresh_Pop, thre_populationGray, thread_num));
	//threshold(ABS_Grad_X, Grad_X_Thresh_Pop, thre_populationGray, 255,THRESH_BINARY);
	delete [] hist;
	//delete cu_hist;
	delete [] min_val_cand;
	delete [] max_val_cand;
}

inline void Y_DirectedGradientGeneration(const Mat &Gray_openingGaussian , Mat &Grad_Y_Thresh_Pop){
	Mat Grad_Y;		
	Mat ABS_Grad_Y = Mat::zeros(Gray_openingGaussian.size()  ,CV_8UC1);
	Mat ScharrY_tmp = Mat::zeros(Gray_openingGaussian.size()  ,CV_16S);
	Mat ConvertScaleAbsY_tmp = Mat::zeros(Gray_openingGaussian.size()  ,CV_8UC1);
	Mat ABS_Grad_YHisto;
	int *hist = new int[256]();
	std::mutex mtx;
	float goal_percentGradthreshold = 0.1*FRAMEW*FRAMEH;
	int thre_populationGray;
	int sum_populationGray = 0;
	double *min_val_cand = new double[thread_num];
	double *max_val_cand = new double[thread_num];
	double min_val = FLT_MAX;
	double max_val = -FLT_MAX;

	//Y Directed Gradient
	//Scharr( Gray_openingGaussian, Grad_Y, CV_16S, 0, 1, 3, 0, BORDER_DEFAULT );
	//convertScaleAbs( Grad_Y, ABS_Grad_Y );
	//normalize(ABS_Grad_Y, ABS_Grad_Y, 0, 255, NORM_MINMAX, CV_8UC1);//L
	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_scharrY(Gray_openingGaussian, ScharrY_tmp, thread_num));
	//cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_convertScaleAbs(ScharrY_tmp, ConvertScaleAbsY_tmp, thread_num));
	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_convertScaleAbs_findMinMax(ScharrY_tmp, ConvertScaleAbsY_tmp, min_val_cand, max_val_cand, thread_num));
	for (int i = 0; i < thread_num; ++i) {
		if (min_val > min_val_cand[i]) {
			min_val = min_val_cand[i];
		}
		if (max_val < max_val_cand[i]) {
			max_val = max_val_cand[i];
		}
	}

	//cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_normalize(ConvertScaleAbsY_tmp, ABS_Grad_Y, thread_num));
	//normalize(ConvertScaleAbsY_tmp, ABS_Grad_Y, 0, 255, NORM_MINMAX, CV_8UC1);//L
	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_converto_min_max(ConvertScaleAbsY_tmp, ABS_Grad_Y, min_val, max_val, 0, 255, thread_num));
		
	//Histogram Grad_Y
	//CalcHistogram(ABS_Grad_Y , ABS_Grad_YHisto);	
	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_hist_pure(ABS_Grad_Y, hist, mtx, thread_num));

	//for(int i=ABS_Grad_YHisto.rows-1;i>0;--i){
	//	sum_populationGray+=ABS_Grad_YHisto.at<float>(0 , i);
	//	if(sum_populationGray>goal_percentGradthreshold){
	//		thre_populationGray = i;
	//		break;
	//	}
	//}
	for (int i = 255; i>0; --i) {
		sum_populationGray += hist[i];
		if (sum_populationGray>goal_percentGradthreshold) {
			thre_populationGray = i;
			break;
		}
	}

	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_threBinary(ABS_Grad_Y, Grad_Y_Thresh_Pop, thre_populationGray, thread_num));
	//threshold(ABS_Grad_Y, Grad_Y_Thresh_Pop, thre_populationGray, 255,THRESH_BINARY);	
	delete [] hist;
	delete [] min_val_cand;
	delete [] max_val_cand;
}

//inline void ConvexHullProcedureFindBounding(const Size &imageSize , const vector<vector<Point> >gradsPts , int &boundary_return
//																	, const bool &left_right){
//	Mat GradsConvexHull = Mat::zeros(imageSize , CV_8UC1);	
//	vector <vector<Point>>  hull(1);			
//	
//	//ConvexHull
//	convexHull( Mat(gradsPts), hull[0], false);		
//	drawContours( GradsConvexHull, hull, 0, Scalar(255 , 255 , 255), CV_FILLED);//Fill the range in the convex hull	
//	Morphology_Operations(GradsConvexHull, GradsConvexHull , MORPH_ERODE, 8,  MORPH_CROSS);	
//
//	//Find leftest(left_right==0) pts or rightest(left_right==1) pts
//	if(left_right==true){
//		int rightestPos = -INT_MAX;
//
//	}else{
//	
//	
//	}
//}

inline void PeakRefinement(const vector<int> &originalPeakDeriv_vx , const vector<vector<Point>> &contoursGrads
										, vector<int> &refinedPeakDeriv_vx)
{
	for(int i=0;i<originalPeakDeriv_vx.size();++i){
		bool hasGrads = false;
		for(int j=0;j<contoursGrads.size();++j){
			int max_x = -INT_MAX;
			int min_x = INT_MAX;
			for(int k=0;k<contoursGrads[j].size();++k){
				if(contoursGrads[j][k].x>max_x){
					max_x = contoursGrads[j][k].x;
				}
				if(contoursGrads[j][k].x<min_x){
					min_x = contoursGrads[j][k].x;
				}
			}
			if(originalPeakDeriv_vx[i]<max_x && originalPeakDeriv_vx[i]>min_x){
				hasGrads = true;
			}
		}
		if(hasGrads){
			refinedPeakDeriv_vx.push_back(originalPeakDeriv_vx[i]);
		}
	}
}

inline void LineFilterRefinement(const vector<vector<Point> > &contours , vector<vector<Point> > &refined_contours
												 , const Point &lineFilterCenter , Mat &RefinedGradX_EyeRegion
												 , vector<Point> &refinedConnectedPoints , Mat &IrisContour
												 , const bool &extremeRight 
												 , const bool &extremeDown
												 , const bool &extremeLeft
												 , const bool &extremeUp
												 , const Point&eyeCoarseCenter											
												 , const vector<int> &leftPeakDeriv_vx , const vector<int> &rightPeakDeriv_vx
												 , const vector<float> &vec_Deriv_GPFv_x
												 , const bool &caculateIris_Mask_done , const bool &getEyeCoarseCenter 
												 , const vector<Point> &IrisContoursPoints_byColor
												 , const int &iris_y_regionUp , const int &iris_y_regionDown
												 , const int &iris_x_regionLeft , const int &iris_x_regionRight)
{		
	IrisContour = Mat::zeros(RefinedGradX_EyeRegion.size() , CV_8UC1);	
	Point gpfCenter;
	float irisMask_SideLength;

	for(int i = 0;i<contours.size();++i){
		int max_y = -INT_MAX;
		int min_y = INT_MAX;
		for(int j=0;j<contours[i].size();++j){
			if(contours[i][j].y>max_y){
				max_y = contours[i][j].y;
			}
			if(contours[i][j].y<min_y){
				min_y = contours[i][j].y;
			}
		}

		if(min_y<lineFilterCenter.y && max_y>lineFilterCenter.y){
			refined_contours.push_back(contours[i]);
		}
	}

	if( !refined_contours.empty()){		
		for( int idx = 0; idx<refined_contours.size();++idx){			
			drawContours(RefinedGradX_EyeRegion , refined_contours, idx, Scalar(255 , 255 , 255), cv::FILLED, 8);
		}
	}	


	irisMask_SideLength = (MIN(iris_x_regionRight - iris_x_regionLeft , iris_y_regionDown - iris_y_regionUp))/2.f*iris_GPF_rectSidePercent;//origin	
	gpfCenter.x = (iris_x_regionRight+iris_x_regionLeft)/2;
	gpfCenter.y = (iris_y_regionUp+iris_y_regionDown)/2;

	circle(IrisContour ,  gpfCenter, irisMask_SideLength, Scalar::all(255) , cv::FILLED);
	RefinedGradX_EyeRegion = RefinedGradX_EyeRegion + IrisContour;	

	for(int i=0;i<RefinedGradX_EyeRegion.rows;++i){
		for(int j=0;j<RefinedGradX_EyeRegion.cols;++j){
			if(RefinedGradX_EyeRegion.at<uchar>(i , j)==255){
				refinedConnectedPoints.push_back(Point(j , i));
			}
		}
	}
}


inline void EyeInnerRegionGeneration(const Mat &Grad_X_Thresh_Pop , const Mat &Grad_Y_Thresh_Pop 
														, const Point &eyeRegionCenter , Mat &ValleyPeakField
														, const Mat &Gray_openingGaussian_Gau_inv, Mat &IrisContour
														, Point &lineFilterCenter , const bool &extremeRight
														, const bool &extremeLeft
														, const bool &extremeDown 
														, const bool &extremeUp
														, const Point &eyeCoarseCenter														
														, const vector<int> &leftPeakDeriv_vx , const vector<int> &rightPeakDeriv_vx
														, const vector<float> &vec_Deriv_GPFv_x
														, const bool &caculateIris_Mask_done , const bool &getEyeCoarseCenter 
														, const vector<Point> &IrisContoursPoints_byColor
														, const int &iris_y_regionUp , const int &iris_y_regionDown
														, const int &iris_x_regionLeft , const int &iris_x_regionRight)
{
	double time_stamp_in0;
	double time_stamp_in1;
	double time_stamp_in2;
	double time_stamp_in3;
	double time_stamp_in4;
	double time_stamp_in5;
	double time_stamp_in6;

	double time_stamp_in1_1;
	double time_stamp_in1_2;
	double time_stamp_in1_3;
	double time_stamp_in1_4;
	double time_stamp_in1_5;

	double time_stamp_in3_1;
	double time_stamp_in3_2;
	double time_stamp_in3_3;
	double time_stamp_in3_4;

	//double time_stamp_in3_core_start;
	//double time_stamp_in3_core_end;

	time_stamp_in0 = getTickCount();

	Mat Grad_X_Y_And = Mat::zeros(Grad_X_Thresh_Pop.size(), CV_8UC1);	
	Mat Grad_X_Minus_And = Mat::zeros(Grad_X_Thresh_Pop.size(), CV_8UC1);
	Mat Grad_X_Y_And_Opening = Mat::zeros(Grad_X_Thresh_Pop.size(), CV_8UC1);
	Mat Grad_X_Y_And_Opening_tmp = Mat::zeros(Grad_X_Thresh_Pop.size(), CV_8UC1);
	Mat Grad_X_Y_And_Opening_Dila = Mat::zeros(Grad_X_Thresh_Pop.size(), CV_8UC1);
	Mat Grad_X_Y_And_Opening_Med = Mat::zeros(Grad_X_Thresh_Pop.size(), CV_8UC1);
	Mat FilterOutNoise_Grad_X = Mat::zeros(Grad_X_Thresh_Pop.size(), CV_8UC1);
	Mat FilterOutNoise_Grad_X_Opening = Mat::zeros(Grad_X_Thresh_Pop.size(), CV_8UC1);
	Mat FilterOutNoise_Grad_X_Opening_tmp = Mat::zeros(Grad_X_Thresh_Pop.size(), CV_8UC1);
	Mat FilterOutNoise_Grad_X_Erosion = Mat::zeros(Grad_X_Thresh_Pop.size(), CV_8UC1);
	Mat ValleyPeakField_tmp = Mat::zeros(Grad_X_Thresh_Pop.size(), CV_8UC1);
	Mat EyelidRegionLast;
	Mat RefinedGradX_EyeRegion = Mat::zeros(Grad_X_Thresh_Pop.size() , CV_8UC1);	
	Mat ContourALL;	
	vector<vector<Point> > contours;
	vector<vector<Point> > refined_contours;
	vector<Point> refinedConnectedPoints;
	Mat morph_cross_element1 = getStructuringElement(MORPH_CROSS, Size(2 * 5 + 1, 2 * 5 + 1), Point(5, 5));
	Mat morph_cross_element2 = getStructuringElement(MORPH_CROSS, Size(2 * 2 + 1, 2 * 2 + 1), Point(2, 2));
	Mat morph_cross_element3 = getStructuringElement(MORPH_CROSS, Size(2 * 8 + 1, 2 * 8 + 1), Point(8, 8));

	//Morphological Operation to Find lineFilterCenter & Grad_X_Y_And_Opening_Dila	
	/*bitwise_and(Grad_X_Thresh_Pop , Grad_Y_Thresh_Pop, Grad_X_Y_And);	
	Grad_X_Minus_And = Grad_X_Thresh_Pop - Grad_X_Y_And;
	Morphology_Operations(Grad_X_Minus_And, Grad_X_Y_And_Opening , MORPH_OPEN, 5,  MORPH_CROSS);				
	Morphology_Operations(Grad_X_Y_And_Opening, Grad_X_Y_And_Opening_Dila , MORPH_DILATE, 5,  MORPH_CROSS);		*/

	time_stamp_in1 = getTickCount();

	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_bitwiseand(Grad_X_Thresh_Pop, Grad_Y_Thresh_Pop, Grad_X_Y_And, thread_num));
	time_stamp_in1_1 = getTickCount();

	//Grad_X_Minus_And = Grad_X_Thresh_Pop - Grad_X_Y_And;
	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_sub(Grad_X_Thresh_Pop, Grad_X_Y_And, Grad_X_Minus_And, thread_num));
	time_stamp_in1_2 = getTickCount();

	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_er(Grad_X_Minus_And, Grad_X_Y_And_Opening_tmp , morph_cross_element1, thread_num));
	time_stamp_in1_3 = getTickCount();

	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_di(Grad_X_Y_And_Opening_tmp, Grad_X_Y_And_Opening, morph_cross_element1, thread_num));
	time_stamp_in1_4 = getTickCount();

	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_di(Grad_X_Y_And_Opening, Grad_X_Y_And_Opening_Dila, morph_cross_element1, thread_num));
	time_stamp_in1_5 = getTickCount();

	

	time_stamp_in2 = getTickCount();

//------------------------------GPU Testing-----------------------------------//
	//cv::cuda::GpuMat dst, src1, src2;

	//double time_upload_start = getTickCount();
	//src1.upload(Gray_openingGaussian_Gau_inv);

	//double time_upload_start1 = getTickCount();
	//src2.upload(Grad_X_Y_And_Opening_Dila);


	//double time_stamp_in3_core_start = getTickCount();

	//cv::cuda::bitwise_and(src1, src2, dst);

	//double time_stamp_in3_core_end = getTickCount();

	//dst.download(FilterOutNoise_Grad_X);

	//double time_download = getTickCount();

	//std::cout << "--------------------bitwise_and begin------------------" << std::endl;
	//std::cout << std::endl << "gpu_upload1 = " << (time_upload_start1 - time_upload_start) / getTickFrequency()<<std::endl;
	//std::cout << std::endl << "gpu_upload2 = " << (time_stamp_in3_core_start - time_upload_start1) / getTickFrequency() << std::endl;
	//std::cout << std::endl << "gpu_bitwise_and = " << (time_stamp_in3_core_end - time_stamp_in3_core_start) / getTickFrequency() << std::endl;
	//std::cout << std::endl << "gpu_download = " << (time_download - time_stamp_in3_core_end) / getTickFrequency() << std::endl;


//	double time_cpu_start = getTickCount();
	//Determine center line
	//bitwise_and(Gray_openingGaussian_Gau_inv , Grad_X_Y_And_Opening_Dila, FilterOutNoise_Grad_X);	
	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_bitwiseand(Gray_openingGaussian_Gau_inv
		, Grad_X_Y_And_Opening_Dila, FilterOutNoise_Grad_X, thread_num));

	time_stamp_in3 = getTickCount();
	//double time_cpu_end = getTickCount();

	//std::cout << std::endl << "cpu_bitwise_and = " << (time_cpu_end - time_cpu_start) / getTickFrequency() << std::endl;
	//std::cout << "--------------------bitwise_and end------------------" << std::endl;

	//------------------------------GPU Testing End-----------------------------------//

	//Morphology_Operations(FilterOutNoise_Grad_X, FilterOutNoise_Grad_X_Opening , MORPH_OPEN, 2,  MORPH_CROSS);		
	//Morphology_Operations(FilterOutNoise_Grad_X_Opening, FilterOutNoise_Grad_X_Erosion , MORPH_ERODE, 2,  MORPH_CROSS);

	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_er(FilterOutNoise_Grad_X, FilterOutNoise_Grad_X_Opening_tmp , morph_cross_element2, thread_num));
	time_stamp_in3_1 = getTickCount();

	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_di(FilterOutNoise_Grad_X_Opening_tmp, FilterOutNoise_Grad_X_Opening, morph_cross_element2, thread_num));
	time_stamp_in3_2 = getTickCount();

	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_er(FilterOutNoise_Grad_X_Opening, FilterOutNoise_Grad_X_Erosion, morph_cross_element2, thread_num));
	time_stamp_in3_3 = getTickCount();

	//CenterCalculatUsingMoment(FilterOutNoise_Grad_X_Erosion , lineFilterCenter.x , lineFilterCenter.y);	
	//CenterCalculatUsingMomentSerial(FilterOutNoise_Grad_X_Erosion , lineFilterCenter.x , lineFilterCenter.y);	
	CenterCalculatUsingMomentParallel(FilterOutNoise_Grad_X_Erosion, lineFilterCenter.x, lineFilterCenter.y, thread_num);
	time_stamp_in3_4 = getTickCount();
	lineFilterCenter.x = (lineFilterCenter.x+eyeRegionCenter.x)/2;
	lineFilterCenter.y = (lineFilterCenter.y+eyeRegionCenter.y)/2;	


	time_stamp_in4 = getTickCount();

	//Line Filter
	FindALLContours(Grad_X_Y_And_Opening_Dila  ,/* ContourALL , */contours);
	LineFilterRefinement(contours , refined_contours , lineFilterCenter , RefinedGradX_EyeRegion 
									, refinedConnectedPoints , IrisContour 
									, extremeRight	
									, extremeDown 	
									, extremeLeft
									, extremeUp
									, eyeCoarseCenter								
									, leftPeakDeriv_vx , rightPeakDeriv_vx
									, vec_Deriv_GPFv_x
									, caculateIris_Mask_done , getEyeCoarseCenter 
									, IrisContoursPoints_byColor
									, iris_y_regionUp , iris_y_regionDown
									, iris_x_regionLeft , iris_x_regionRight);


	time_stamp_in5 = getTickCount();

	//Convex Hull
	vector <vector<Point>>  hull(1);			
	convexHull( Mat(refinedConnectedPoints), hull[0], false);		
	drawContours( ValleyPeakField_tmp, hull, 0, Scalar(255 , 255 , 255), cv::FILLED);//Fill the range in the convex hull	
	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_di(ValleyPeakField_tmp, ValleyPeakField, morph_cross_element3, thread_num));
	//Morphology_Operations(ValleyPeakField_tmp, ValleyPeakField , MORPH_DILATE, 8,  MORPH_CROSS);			


	time_stamp_in6 = getTickCount();
	time_eye_position_detection_eyelid_feature_detection_district4_0.push_back(time_stamp_in1 - time_stamp_in0);
	time_eye_position_detection_eyelid_feature_detection_district4_1.push_back(time_stamp_in2 - time_stamp_in1);
	time_eye_position_detection_eyelid_feature_detection_district4_2.push_back(time_stamp_in3 - time_stamp_in2);
	//time_eye_position_detection_eyelid_feature_detection_district4_2_core.push_back(time_stamp_in3_core_end - time_stamp_in3_core_start);
	time_eye_position_detection_eyelid_feature_detection_district4_3.push_back(time_stamp_in4 - time_stamp_in3);
	time_eye_position_detection_eyelid_feature_detection_district4_4.push_back(time_stamp_in5 - time_stamp_in4);
	time_eye_position_detection_eyelid_feature_detection_district4_5.push_back(time_stamp_in6 - time_stamp_in5);
	time_eye_position_detection_eyelid_feature_detection_district4_local_sum.push_back(time_stamp_in6 - time_stamp_in0);

	time_eye_position_detection_eyelid_feature_detection_district4_1_1.push_back(time_stamp_in1_1 - time_stamp_in1);
	time_eye_position_detection_eyelid_feature_detection_district4_1_2.push_back(time_stamp_in1_2 - time_stamp_in1_1);
	time_eye_position_detection_eyelid_feature_detection_district4_1_3.push_back(time_stamp_in1_3 - time_stamp_in1_2);
	time_eye_position_detection_eyelid_feature_detection_district4_1_4.push_back(time_stamp_in1_4 - time_stamp_in1_3);
	time_eye_position_detection_eyelid_feature_detection_district4_1_5.push_back(time_stamp_in1_5 - time_stamp_in1_4);

	time_eye_position_detection_eyelid_feature_detection_district4_3_1.push_back(time_stamp_in3_1 - time_stamp_in3);
	time_eye_position_detection_eyelid_feature_detection_district4_3_2.push_back(time_stamp_in3_2 - time_stamp_in3_1);
	time_eye_position_detection_eyelid_feature_detection_district4_3_3.push_back(time_stamp_in3_3 - time_stamp_in3_2);
	time_eye_position_detection_eyelid_feature_detection_district4_3_4.push_back(time_stamp_in3_4 - time_stamp_in3_3);

}

inline void IrisMaskPreProcessing_byIntensityModel(const Mat &IrisContour , Mat &IrisMask 
																			, const Mat &EyelidRegion , const bool &getEyelidRegion
																			, const int &iris_x_regionRight , const int &iris_x_regionLeft
																			, const float * const lowerParabolaTable , const float *const upperParabolaTable
																			, const Point &vertex_upperEyelid , const Point &vertex_lowerEyelid
																			, const Point &eyeCoarseCenter)
{
	Mat IrisMask_cal = Mat::zeros(Frame_wh.size() , CV_8UC1);;
	float vertexVerticalDist = (vertex_lowerEyelid.y - vertex_upperEyelid.y);
	//float shift_eyelidRegionUpper = vertexVerticalDist/8;
	//float shift_eyelidRegionLower = vertexVerticalDist/10;
	//float shift_eyelidRegionLeft = vertexVerticalDist/8;
	//float shift_eyelidRegionRight = vertexVerticalDist/18;
	float shift_eyelidRegionLower = 25000/(vertexVerticalDist+eyeCoarseCenter.y);
	float shift_eyelidRegionUpper = 5600/vertexVerticalDist;
	float shift_eyelidRegionLeft = 15000/(vertexVerticalDist+eyeCoarseCenter.y);
	float shift_eyelidRegionRight = 14000/(vertexVerticalDist+eyeCoarseCenter.y);

	//float shift_eyelidRegionLower = 20000/(vertexVerticalDist+exp(eyeCoarseCenter.y/35.f));
	//float shift_eyelidRegionUpper = 5600/vertexVerticalDist;
	//float shift_eyelidRegionLeft = 15000/(vertexVerticalDist+eyeCoarseCenter.y);
	//float shift_eyelidRegionRight = 14000/(vertexVerticalDist+eyeCoarseCenter.y);



	

	if(getEyelidRegion)
		IrisMask_cal = EyelidRegion.clone();	
	else{
		return;
	}
		
	for(int i=0;i<IrisMask_cal.rows;++i){
		for(int j=0;j<IrisMask_cal.cols;++j){
			if(j<iris_x_regionLeft + shift_eyelidRegionLeft || j>iris_x_regionRight - shift_eyelidRegionRight 
				|| i<upperParabolaTable[j] + shift_eyelidRegionUpper || i>lowerParabolaTable[j] - shift_eyelidRegionLower){
				IrisMask_cal.at<uchar>(i , j) = 0;
			}
		}
	}
	int irisMaskForLimbusErode_sizeCal = countNonZero(IrisMask_cal)/intensityErosionBaseSize_experiment*irisMaskForLimbusErode_size;
	Mat morph_cross_element = getStructuringElement(MORPH_CROSS, Size(2 * irisMaskForLimbusErode_sizeCal + 1, 2 * irisMaskForLimbusErode_sizeCal + 1), Point(irisMaskForLimbusErode_sizeCal, irisMaskForLimbusErode_sizeCal));
	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_er(IrisMask_cal, IrisMask , morph_cross_element, thread_num));
	//Morphology_Operations(IrisMask_cal, IrisMask , MORPH_ERODE, irisMaskForLimbusErode_sizeCal,  MORPH_CROSS);	
}





inline void VPFh_yCaculate(const Mat &Src , const Mat &IPFh_y , Mat &VPFh_y){
	VPFh_y = Mat::zeros(Src.rows , 1 , CV_32FC1);
	for(int i=0;i<Src.rows;++i){
		float var_x = 0;
		for(int j=0;j<Src.cols;++j){
			var_x+= powf((float)Src.at<uchar>(i , j) - IPFh_y.at<float>(i , 0) , 2);
		}
		var_x/=(float)Src.cols;		
		VPFh_y.at<float>(i , 0) = var_x;
	}	
}

inline void VPFv_xCaculate(const Mat &Src , const Mat &IPFv_x , Mat &VPFv_x){
	VPFv_x = Mat::zeros(1 , Src.cols , CV_32FC1);
	for(int i=0;i<Src.cols;++i){
		float var_y = 0;
		for(int j=0;j<Src.rows;++j){
			var_y+= powf((float)Src.at<uchar>(j , i) - IPFv_x.at<float>(0 , i) , 2);
		}
		var_y/=(float)Src.rows;
		VPFv_x.at<float>(0 , i) = var_y;
	}	
}
inline void ConvertMatToVec1D(const Mat &src , vector<float> &outData ,const bool ISColumnVec){
	if(ISColumnVec){				
		for(int i=0;i<src.rows;++i){
			outData.push_back(src.at<float>(i , 0));
		}
	}else{		
		for(int i=0;i<src.cols;++i){
			outData.push_back(src.at<float>(0 , i));
		}
	}
}
inline void ConvertMatToVec1D_ver2(const Mat &src , vector<float> &outData){
	src.copyTo(outData);
}

inline void FindPeaks(const vector<float> &input , vector<int> &findout , const int &comparing_size){
	for(int i=1;i<input.size()-1;++i){
		bool find = true;
		if(i==0){
			continue;
		}else if(i==input.size()-1){
			continue;
		}else{
			for(int j=0;j<comparing_size;++j){
				if(i-j-1<0 || i+1+j>input.size()-1)break;
				
				float left_value = input[i-j-1];
				float right_value = input[i+1+j];
				if(input[i]>=left_value && input[i]>=right_value){
					continue;
				}else{
					find = false;
				}
			}

			if(!find){
				continue;
			}else{
				findout.push_back(i);
			}
		}
	}
}

inline void FindPeaks_ver2(const vector<float> &input , vector<int> &findout , const int &comparing_size){
	for(int i=0;i<input.size();++i){
		bool find_left = true;
		bool find_right = true;
		if(i==0){
			continue;
		}else if(i==input.size()-1){
			continue;
		}else{
			for(int j=0;j<comparing_size;++j){
				if(i-j-1<0)break;
				
				float left_value = input[i-j-1];				
				if(input[i]>=left_value){
					continue;
				}else{
					find_left = false;
				}
			}
			for(int j=0;j<comparing_size;++j){
				if(i+1+j>input.size()-1)break;
				
				float right_value = input[i+1+j];			
				if(input[i]>=right_value){
					continue;
				}else{
					find_right = false;
				}
			}



			if(!find_left || !find_right){
				continue;
			}else{
				findout.push_back(i);
			}
		}
	}
}


inline void FindValley(const vector<float> &input , vector<int> &findout , const int &comparing_size){
	for(int i=0;i<input.size();++i){
		bool find = true;
		if(i==0){
			continue;
		}else if(i==input.size()-1){
			continue;
		}else{
			for(int j=0;j<comparing_size;++j){
				if(i-j-1<0 || i+1+j>input.size()-1)break;

				float left_value = input[i-j-1];
				float right_value = input[i+1+j];
				if(input[i]<=left_value && input[i]<=right_value){
					continue;
				}else{
					find = false				;
				}
			}

			if(!find){
				continue;
			}else{
				findout.push_back(i);
			}
		}
	}
}


//inline void Derivative_1D(const vector<float> &input , vector<float> &deriv_out , const int &scale){
//	for(int i=0;i<input.size();++i){
//		if(i+scale>input.size()-1)break;
//		float sumofDeriv = 0;
//		for(int j=i+1;j<i+scale+1;++j){
//			sumofDeriv+=fabs(input[i] - input[j]);
//		}
//		sumofDeriv/=scale;
//		deriv_out.push_back(sumofDeriv);
//	}
//}

inline void Derivative_1D(const vector<float> &input , vector<float> &deriv_out , const int &scale){
	for(int i=0;i<input.size();++i){
		//if(i+scale>input.size()-1)break;
		float sumofDeriv = 0;

		if (i + scale > input.size() - 1) {
			for (int j = i + 1; j<input.size(); ++j) {
				sumofDeriv += fabs(input[i] - input[j]);
			}
		}
		else {
			for (int j = i + 1; j < i + scale + 1; ++j) {
				sumofDeriv += fabs(input[i] - input[j]);
			}
		}
		sumofDeriv/=scale;
		deriv_out.push_back(sumofDeriv);
	}
}


inline bool GeneralProjectionFunction_Ver2(const Mat &Src , Point &eyeCoarseCenter 												 
																 , int &iris_x_regionRight , int &iris_x_regionLeft
																 , int &iris_y_regionUp , int &iris_y_regionDown
																 , vector<int> &leftPeakDeriv_vx , vector<int> &rightPeakDeriv_vx
																 , vector<float> &vec_Deriv_GPFv_x)
{
	double time_start1 = getTickCount();
	Mat IPFh_y;	
	Mat IPFv_x;
	Mat VPFh_y;
	Mat VPFv_x;
	Mat GPFh_y = Mat::zeros(Src.rows , 1 , CV_32FC1);
	Mat GPFv_x = Mat::zeros(1 , Src.cols , CV_32FC1);
	Mat Src_forX;
	vector<float> vec_GPFh_y;
	vector<float> vec_GPFv_x;
	vector<int> find_hy_peak;
	vector<int> find_vx_peak;
	vector<int> find_hy_Derivpeak;
	vector<int> find_vx_Derivpeak;
	vector<float> vec_Deriv_GPFh_y;		
	double max_varh_y;
	double max_varv_x;
	float alpha = 0;		
	Rect Region_of_interest;


	//==========================Process Y First============================/
	//IPF
	reduce(Src, IPFh_y, 1, cv::REDUCE_AVG , CV_32FC1);//IPFh(y)		

	//VPF
	//VPFh_yCaculate(Src , IPFh_y , VPFh_y);	

	//Normalize
	//minMaxLoc(VPFh_y, 0, &max_varh_y);
	GPFh_y = IPFh_y.clone();
	//for(int i=0;i<GPFh_y.rows;++i){
	//	GPFh_y.at<float>(i , 0) = (1-alpha)*IPFh_y.at<float>(i , 0) + alpha*VPFh_y.at<float>(i , 0)/max_varh_y*255;
	//}

	//Derivative
	GPFh_y.copyTo(vec_GPFh_y);//ConvertMatToVec1D_ver2
	Derivative_1D(vec_GPFh_y , vec_Deriv_GPFh_y , 10);
		
	
	//Normalize	
	double max_deriv_hy;
	minMaxLoc(vec_Deriv_GPFh_y, 0, &max_deriv_hy);	
	transform( vec_Deriv_GPFh_y.begin(), vec_Deriv_GPFh_y.end()
					, vec_Deriv_GPFh_y.begin(), bind1st( multiplies<float>(), 1.f/(max_deriv_hy)*255));

	//for(int i=0;i<vec_Deriv_GPFh_y.size();++i){
	//	vec_Deriv_GPFh_y[i] = vec_Deriv_GPFh_y[i]/max_deriv_hy*255;
	//}

	//FindPeaks
	//FindPeaks(vec_Deriv_GPFh_y , find_hy_Derivpeak , 100);
	FindPeaks_ver2(vec_Deriv_GPFh_y , find_hy_Derivpeak , 100);
	

	if(find_hy_Derivpeak.size()<2){
		printf("Didn't get proper IrisContour_byIntensity\n");		
		return false;
	}
	//Sorting to Get Two Largest Derivative of hy
	FindPeakof1D_Data* hy_Deriv_peakDataAndPos = new FindPeakof1D_Data [find_hy_Derivpeak.size()];
	for(int i=0;i<find_hy_Derivpeak.size();++i){
		hy_Deriv_peakDataAndPos[i].element = vec_Deriv_GPFh_y[find_hy_Derivpeak[i]];
		hy_Deriv_peakDataAndPos[i].number = find_hy_Derivpeak[i];
	}
	sort(hy_Deriv_peakDataAndPos , hy_Deriv_peakDataAndPos+find_hy_Derivpeak.size());

	iris_y_regionUp = hy_Deriv_peakDataAndPos[find_hy_Derivpeak.size()-1].number;
	iris_y_regionDown = hy_Deriv_peakDataAndPos[find_hy_Derivpeak.size()-2].number;

	if(iris_y_regionUp>iris_y_regionDown)
		swap(iris_y_regionUp , iris_y_regionDown);

	//==========================Set ROI============================/	
	Region_of_interest = Rect(0, iris_y_regionUp, Src.cols , iris_y_regionDown - iris_y_regionUp);
	Src_forX = Src(Region_of_interest);	
	

	//==========================Process X Last============================/
	//IPF	
	reduce(Src_forX, IPFv_x, 0, cv::REDUCE_AVG , CV_32FC1);//IPFv(x)

	//VPF	
	//VPFv_xCaculate(Src_forX , IPFv_x , VPFv_x);

	//Normalise
	//minMaxLoc(VPFv_x, 0, &max_varv_x);	
	GPFv_x = IPFv_x.clone();
	//for(int i=0;i<GPFv_x.cols;++i){
	//	GPFv_x.at<float>(0 , i) = IPFv_x.at<float>(0 , i);
	//}
	//for(int i=0;i<GPFv_x.cols;++i){
	//	GPFv_x.at<float>(0 , i) = (1-alpha)*IPFv_x.at<float>(0 , i) + alpha*VPFv_x.at<float>(0 , i)/max_varv_x*255;
	//}
			

	//Derivative	
	GPFv_x.copyTo(vec_GPFv_x);//ConvertMatToVec1D_ver2
	Derivative_1D(vec_GPFv_x , vec_Deriv_GPFv_x , 10);	
	
	

	//Normalise
	double max_deriv_vx;	
	minMaxLoc(vec_Deriv_GPFv_x, 0, &max_deriv_vx);
	transform( vec_Deriv_GPFv_x.begin(), vec_Deriv_GPFv_x.end()
					, vec_Deriv_GPFv_x.begin(), bind1st( multiplies<float>(), 1.f/(max_deriv_vx)*255));
	/*for(int i=0;i<vec_Deriv_GPFv_x.size();++i){
		vec_Deriv_GPFv_x[i] = vec_Deriv_GPFv_x[i]/max_deriv_vx*255;
	}*/
	
	//FindPeaks	
	FindPeaks_ver2(vec_Deriv_GPFv_x , find_vx_Derivpeak , 100);

	time_eye_position_detection_iris_region_extraction.push_back(getTickCount() - time_start1);
	//==========================Get Coarse Eye Region Mask============================/	
	for(int i=0;i<find_vx_Derivpeak.size();++i){
		if(find_vx_Derivpeak[i]<eyeCoarseCenter.x){
			leftPeakDeriv_vx.push_back(find_vx_Derivpeak[i]);
		}else{
			rightPeakDeriv_vx.push_back(find_vx_Derivpeak[i]);
		}
	}

	if(leftPeakDeriv_vx.size()==0){
		iris_x_regionLeft = 0;	
	}else{
		int firstHighest;
		int secondHighest;
		FindPeakof1D_Data* vx_LeftDeriv_peakDataAndPos = new FindPeakof1D_Data [leftPeakDeriv_vx.size()];
		for(int i=0;i<leftPeakDeriv_vx.size();++i){
			vx_LeftDeriv_peakDataAndPos[i].element = vec_Deriv_GPFv_x[leftPeakDeriv_vx[i]];
			vx_LeftDeriv_peakDataAndPos[i].number = leftPeakDeriv_vx[i];
		}
		sort(vx_LeftDeriv_peakDataAndPos , vx_LeftDeriv_peakDataAndPos + leftPeakDeriv_vx.size());

		firstHighest = vx_LeftDeriv_peakDataAndPos[leftPeakDeriv_vx.size()-1].element;
		secondHighest = vx_LeftDeriv_peakDataAndPos[leftPeakDeriv_vx.size()-2].element;

		if(firstHighest>200 && secondHighest>200){
			if(vx_LeftDeriv_peakDataAndPos[leftPeakDeriv_vx.size()-1].number < vx_LeftDeriv_peakDataAndPos[leftPeakDeriv_vx.size()-2].number){
				iris_x_regionLeft = vx_LeftDeriv_peakDataAndPos[leftPeakDeriv_vx.size()-2].number;
			}else{
				iris_x_regionLeft = vx_LeftDeriv_peakDataAndPos[leftPeakDeriv_vx.size()-1].number;
			}
		}else{
			iris_x_regionLeft = vx_LeftDeriv_peakDataAndPos[leftPeakDeriv_vx.size()-1].number;		
		}	

		delete [] vx_LeftDeriv_peakDataAndPos;
	}



	if(rightPeakDeriv_vx.size()==0){
		iris_x_regionRight = Src.cols - 1;
	}else if(rightPeakDeriv_vx.size()==1){
		iris_x_regionRight = rightPeakDeriv_vx[0];
	}else{
		int firstHighest;
		int secondHighest;
		FindPeakof1D_Data* vx_RightDeriv_peakDataAndPos = new FindPeakof1D_Data [rightPeakDeriv_vx.size()];
		for(int i=0;i<rightPeakDeriv_vx.size();++i){
			vx_RightDeriv_peakDataAndPos[i].element = vec_Deriv_GPFv_x[rightPeakDeriv_vx[i]];
			vx_RightDeriv_peakDataAndPos[i].number = rightPeakDeriv_vx[i];
		}
		sort(vx_RightDeriv_peakDataAndPos , vx_RightDeriv_peakDataAndPos + rightPeakDeriv_vx.size());	
		firstHighest = vx_RightDeriv_peakDataAndPos[rightPeakDeriv_vx.size()-1].element;
		secondHighest = vx_RightDeriv_peakDataAndPos[rightPeakDeriv_vx.size()-2].element;

		if(firstHighest>200 && secondHighest>200){
			if(vx_RightDeriv_peakDataAndPos[rightPeakDeriv_vx.size()-1].number < vx_RightDeriv_peakDataAndPos[rightPeakDeriv_vx.size()-2].number){
				iris_x_regionRight = vx_RightDeriv_peakDataAndPos[rightPeakDeriv_vx.size()-1].number;
			}else{
				iris_x_regionRight = vx_RightDeriv_peakDataAndPos[rightPeakDeriv_vx.size()-2].number;
			}
		}else{			
			iris_x_regionRight = vx_RightDeriv_peakDataAndPos[rightPeakDeriv_vx.size()-1].number;	
		}

		delete [] vx_RightDeriv_peakDataAndPos;
	}	

	delete [] hy_Deriv_peakDataAndPos;
	return true;
}


#pragma endregion

#pragma region

inline bool InitialCoarseCenterCDF(const Mat &Src , Point &irisCoarseCenter){
	int histBins = 256;
	float cumulativeNumber = 0;
	vector<Point> IrisContoursPoints;
	Mat GrayGaussian;
	Mat GrayOtsu;				
	Mat Src_Binary = Mat::zeros(Src.size() , CV_8UC1);
	Mat Src_Hist;
	Mat Hist_Sum;
	Mat Hist_CDF = Mat::zeros(1 , histBins , CV_32FC1);	
	Mat IrisContour;

	int *cu_hist = new int[256]();
	std::mutex mtx;
	int thresh_cdf;
	int total_size = Src.rows * Src.cols;

	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_hist_and_cumulative_pure(Src, cu_hist, mtx, thread_num));

	double time_serial_start1 = getTickCount();
	for (int k = 0; k < 256; ++k) {
		if (cu_hist[k] / (double)total_size > 0.05) {
			thresh_cdf = k;
			break;
		}
	}
	double time_serial_end1 = getTickCount();

	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_apply_threshold(Src, Src_Binary, thresh_cdf-1, cv::THRESH_BINARY_INV, thread_num));


	//imshow("pre", Src);
	//imshow("aft", Src_Binary);
	//waitKey(0);
	//Probability Density Function (PDF)
	//CalcHistogram(Src , Src_Hist);		
	//reduce(Src_Hist, Hist_Sum, 0 ,  cv::REDUCE_SUM);//SUM
	//for(int i=0;i<Src_Hist.rows;++i){		
	//	Src_Hist.at<float>(i , 0) /= Hist_Sum.at<float>(0 , 0);
	//}
	//
	////Cumulative Density Function (CDF)
	//for(int i=0;i<Hist_CDF.cols;++i){
	//	cumulativeNumber+=Src_Hist.at<float>(i , 0);
	//	Hist_CDF.at<float>(0 , i)=cumulativeNumber;
	//}
	//
	//int thresh_cdf_serial = -FLT_MAX;
	//for(int i=0;i<Src.rows;++i){
	//	for(int j=0;j<Src.cols;++j){
	//		int intensity = Src.at<uchar>(i , j);
	//		if(Hist_CDF.at<float>(0 , intensity)<0.05){
	//			if (intensity > thresh_cdf_serial) {
	//				thresh_cdf_serial = intensity;
	//			}
	//			Src_Binary.at<uchar>(i , j) = 255;
	//		}
	//	}
	//}

	//std::cout << "thresh_cdf_serial = " << thresh_cdf_serial << std::endl;
	
	double time_serial_start2 = getTickCount();
	if(!FindMAXConnextedComponent(Src_Binary , IrisContoursPoints , IrisContour)){
		return false;
	}
	double time_serial_end2 = getTickCount();
	time_eye_position_detection_coarse_iris_location_cdf_serial.push_back((time_serial_end2 - time_serial_start2)+(time_serial_end1 - time_serial_start1));
	//std::cout << "FindMAXConnectedComponent takes time = " << (double)(time_2 - time_1) / getTickFrequency() << std::endl;
	//CenterCalculatUsingMoment(IrisContoursPoints , irisCoarseCenter.x , irisCoarseCenter.y);
	//std::cout << "OpenCV memoent = (" << irisCoarseCenter.x << ", " << irisCoarseCenter.y << ")" << std::endl;
	CenterCalculatUsingMomentParallel(IrisContoursPoints, irisCoarseCenter.x, irisCoarseCenter.y, thread_num);
	//std::cout << "CenterCalculatUsingMomentParallel memoent = (" << irisCoarseCenter.x << ", " << irisCoarseCenter.y << ")" << std::endl;

	//imshow("Src", Src);
	//imshow("Src_Binary", Src_Binary);
	//waitKey(0);

	return true;
}

inline void ClipFeaturePoints(const vector<Point> feature_point_Limbus , vector<Point> &clipped_feature_point_Limbus
											, const int &iris_x_regionRight , const int &iris_x_regionLeft
											, const float &dis_thresholdSoft_lowerEyelid , const float &dis_thresholdSoft_upperEyelid								    
											, const float *lower_parabola_param , const float *upper_parabola_param
											, const bool &getEyelidRegion , const Mat &EyelidRegion
											, vector<Point> &upperNoiseFts_Clipped , vector<Point> &lowerNoiseFts_Clipped)
{
	Mat ClipFeaturePointsDisp = Frame_wh.clone();	
	float dis_thresholdSoft_lowerEyelid_LimbusRefined = dis_thresholdSoft_lowerEyelid*2;
	float dis_thresholdSoft_upperEyelid_LimbusRefined = dis_thresholdSoft_upperEyelid*1.5;


	for(int i=0;i<feature_point_Limbus.size();++i){
		float err_LowPar =  fabs(ErrorParabolaEOF2(lower_parabola_param , feature_point_Limbus[i]));		
		float err_UpPar =  fabs(ErrorParabolaEOF2(upper_parabola_param , feature_point_Limbus[i]));
		if(err_LowPar<dis_thresholdSoft_lowerEyelid_LimbusRefined){			
			if(err_UpPar<dis_thresholdSoft_upperEyelid_LimbusRefined){
				upperNoiseFts_Clipped.push_back(feature_point_Limbus[i]);							
			}else{				
				lowerNoiseFts_Clipped.push_back(feature_point_Limbus[i]);
			}
		}else{			
			if(err_UpPar<dis_thresholdSoft_upperEyelid_LimbusRefined){						
				upperNoiseFts_Clipped.push_back(feature_point_Limbus[i]);
			}else{
				if(getEyelidRegion){
					if(EyelidRegion.at<uchar>(feature_point_Limbus[i].y , feature_point_Limbus[i].x)==255){
						clipped_feature_point_Limbus.push_back(feature_point_Limbus[i]);												
					}
				}else{
					clipped_feature_point_Limbus.push_back(feature_point_Limbus[i]);										
				}
			}				
		}	
	}
}

inline void DetermineWhetherEyeRefineCenterInExtremeRegion(const Point &eyeRefinedIrisCenter
																								, bool &extremeRight_forBlink
																								, bool &extremeDown_forBlink
																								, bool &extremeUp_forBlink
																								, bool &extremeLeft_forBlink
																								, const float &eyeCoarseIrisCenter_ExtremeRegion_Right
																								, const float &eyeCoarseIrisCenter_ExtremeRegion_Down
																								, const float &eyeCoarseIrisCenter_ExtremeRegion_Up
																								, const float &eyeCoarseIrisCenter_ExtremeRegion_Left)
{
	Mat DisplayExtremeRegion = Frame_wh.clone();
	if(eyeRefinedIrisCenter.x>eyeCoarseIrisCenter_ExtremeRegion_Right){		
		extremeRight_forBlink = true;		
	}
	if(eyeRefinedIrisCenter.x<eyeCoarseIrisCenter_ExtremeRegion_Left){		
		extremeLeft_forBlink = true;		
	}
	
	if(eyeRefinedIrisCenter.y>eyeCoarseIrisCenter_ExtremeRegion_Down){
		extremeDown_forBlink = true;	
	}

	if(eyeRefinedIrisCenter.y<eyeCoarseIrisCenter_ExtremeRegion_Up){
		extremeUp_forBlink = true;
	}	
}

inline void DetermineWhetherEyeCoarseCenterInExtremeRegion(const Point &eyeCoarseIrisCenter
																								, bool &extremeRight
																								, bool &extremeDown
																								, bool &extremeUp
																								, bool &extremeLeft
																								, const float &eyeCoarseIrisCenter_ExtremeRegion_Right
																								, const float &eyeCoarseIrisCenter_ExtremeRegion_Down
																								, const float &eyeCoarseIrisCenter_ExtremeRegion_Up
																								, const float &eyeCoarseIrisCenter_ExtremeRegion_Left)
{
	if(eyeCoarseIrisCenter.x>eyeCoarseIrisCenter_ExtremeRegion_Right){		
		extremeRight = true;		
	}
	if(eyeCoarseIrisCenter.x<eyeCoarseIrisCenter_ExtremeRegion_Left){		
		extremeLeft = true;		
	}
	
	if(eyeCoarseIrisCenter.y>eyeCoarseIrisCenter_ExtremeRegion_Down){
		extremeDown = true;		
	}

	if(eyeCoarseIrisCenter.y<eyeCoarseIrisCenter_ExtremeRegion_Up){
		extremeUp = true;
	}	
}



inline void RefreshExtremeRegionLimit_eyeCoarseCenter(const Point &eyeRefinedIrisCenter
																					 , float &eyeCoarseIrisCenter_ExtremeRegion_Right
																					 , float &eyeCoarseIrisCenter_ExtremeRegion_Down
																					 , float &eyeCoarseIrisCenter_ExtremeRegion_Up
																					 , float &eyeCoarseIrisCenter_ExtremeRegion_Left
																					 , float &eyeCenter_rightestPos
																					 , float &eyeCenter_lowestPos
																					 , float &eyeCenter_uppestPos
																					 , float &eyeCenter_leftestPos)
{
	
	if(eyeRefinedIrisCenter.x>eyeCenter_rightestPos){
		eyeCenter_rightestPos = eyeRefinedIrisCenter.x;
	}
	if(eyeRefinedIrisCenter.x<eyeCenter_leftestPos){
		eyeCenter_leftestPos = eyeRefinedIrisCenter.x;
	}
	if(eyeRefinedIrisCenter.y>eyeCenter_lowestPos){
		eyeCenter_lowestPos =eyeRefinedIrisCenter.y;
	}
	if(eyeRefinedIrisCenter.y<eyeCenter_uppestPos){
		eyeCenter_uppestPos = eyeRefinedIrisCenter.y;		
	}

	eyeCoarseIrisCenter_ExtremeRegion_Right = eyeCenter_rightestPos - 40;
	eyeCoarseIrisCenter_ExtremeRegion_Left = eyeCenter_leftestPos + 40;
	eyeCoarseIrisCenter_ExtremeRegion_Down = eyeCenter_lowestPos - 25;
	eyeCoarseIrisCenter_ExtremeRegion_Up = eyeCenter_uppestPos + 35;//35;
}



inline void MedianFilter1D(const deque<int> &pos_queue , int &pos_return , const int &median_filter1D_size){
	deque<int> ratio_queue_cal(pos_queue);
	sort(ratio_queue_cal.begin() , ratio_queue_cal.end());
	pos_return = ratio_queue_cal[median_filter1D_size/2];
}


inline void IrisBoundaryMedianFilter1D(deque<int> &posQueue , const int &medSize , int &inputPos){
	if(posQueue.size()<medSize){
		posQueue.push_back(inputPos);
	}else{
		posQueue.pop_front();
		posQueue.push_back(inputPos);

		MedianFilter1D(posQueue , inputPos , medSize);
	}
}

inline void MedianFilter1DProcedure(deque<int> &posQueue , const int &medSize , const int &inputPos , int &outPos){
	if(posQueue.size()<medSize){
		posQueue.push_back(inputPos);
		outPos = inputPos;
	}else{
		posQueue.pop_front();
		posQueue.push_back(inputPos);

		MedianFilter1D(posQueue , outPos , medSize);
	}
}

inline void InterFrameFilter(deque<int> &yPos_que , const int &newCenterInput_y , const int &filterFrameSize
										, int &outCenter_y , float &meanLast)
{
	if(yPos_que.size()<filterFrameSize){
		yPos_que.push_back(newCenterInput_y);		
		outCenter_y = (meanLast*(yPos_que.size()-1)+newCenterInput_y)/(float)yPos_que.size();
		meanLast = outCenter_y;
	}else{
		outCenter_y = (yPos_que.size()*meanLast - yPos_que.front()+newCenterInput_y)/(float)yPos_que.size();
		meanLast = outCenter_y;
		yPos_que.pop_front();
		yPos_que.push_back(newCenterInput_y);		
	}	
}



inline void EyePositionDetection(const int frame_number ,const Mat &Frame , const Mat &Frame_Gray 
												, Mat &EyePosition_Result , Mat &EyePosition_CenterResult
												, vector<Point2f> &centerEstimation
												, const Mat &Iris_Mask , const bool &caculateIris_Mask_done
												, Mat &IrisRegionValidTesting
												, const int &countForColorModelHistValidTesting
												, bool &bigmotionIrisCenter , bool &noLimbusFeaturePts
												, Point &vertexParabolaLower , Point &vertexParabolaUpper
												, Point &vertexParabolaUpperFirstFrame , Point &vertexParabolaLowerFirstFrame 
												, vector<Point> &IrisContoursPoints , bool &getIrisContourPoints
												, bool &extremeRight_forBlink , bool &extremeDown_forBlink 
												, bool &extremeUp_forBlink , bool &extremeLeft_forBlink
												, float &irisContour_size
												, float &eyeCloseDetermine_irisContourSizeThreshold_colorModelBased
												, float &eyeCoarseIrisCenter_ExtremeRegion_Right , float &eyeCoarseIrisCenter_ExtremeRegion_Up
												, float &eyeCoarseIrisCenter_ExtremeRegion_Down , float &eyeCoarseIrisCenter_ExtremeRegion_Left
												, float &eyeCenter_rightestPos	, float &eyeCenter_lowestPos
												, float &eyeCenter_uppestPos , float &eyeCenter_leftestPos
												, deque<int> &iris_boundaty_Left , deque<int> &iris_boundaty_Right
												, Point &eyeRefinedIrisCenter , bool &caculateComplete
												, const bool &calibrationProcedureBegin
												, deque<int> &posNonlinearRegionEyeQueueY
												, ofstream &file_time_fpsOut
												, VideoWriter &writer_result_roi)
{	
	
	//if(printDebug){
	//	printf("\n 1\n");
	//}



	//-------------------High Frequency Component Removal---------------------//	
	double time_opening = 0;
	double time_gaussian_blur = 0;
	const int openint_size = 19;
	const int size_gaussian = 61;
	Mat Gray_opening = cv::Mat::zeros(Frame_Gray.size(), CV_8UC1);
	Mat Gray_openingGaussian = cv::Mat::zeros(Frame_Gray.size(), CV_8UC1);
	Mat Tmp_open = cv::Mat::zeros(Frame_Gray.size(), CV_8UC1);
	Mat Tmp_gau = cv::Mat::zeros(Frame_Gray.size(), CV_8UC1);
	Mat morph_rect_element = getStructuringElement(MORPH_RECT, Size(2 * openint_size + 1, 2 * openint_size + 1), Point(openint_size, openint_size));

	
	time_start_in_function = getTickCount();

	//cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_er(Frame_Gray, Tmp_open, openint_size, MORPH_RECT, thread_num));
	//cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_di(Tmp_open, Gray_opening, openint_size, MORPH_RECT, thread_num));
	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_er(Frame_Gray, Tmp_open, morph_rect_element, thread_num));
	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_di(Tmp_open, Gray_opening, morph_rect_element, thread_num));
	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_gau(Gray_opening, Gray_openingGaussian, size_gaussian, thread_num));
	
	time_end_in_function = getTickCount();
	time_image_preprocessing_high_freq_removal.push_back(time_end_in_function - time_start_in_function);
	
	//if(printDebug){
	//	printf("\n 2\n");
	//}

	//-------------------Coarse IRIS Location---------------------//	
	vector<int> leftPeakDeriv_vx;
	vector<int> rightPeakDeriv_vx;
	vector<float> vec_Deriv_GPFv_x;
	bool extremeLeft = false;
	bool extremeRight = false;
	bool extremeUp = false;
	bool extremeDown = false;
	int iris_x_regionRight;
	int iris_x_regionLeft;
	int iris_y_regionUp;
	int iris_y_regionDown;
	Point eyeCoarseCenter(0 , 0);		
	Mat IrisContour_byColor;	
	Mat IrisContour_byIntensity;		
	bool getEyeCoarseCenter = false;
	bool getIrisContour_byIntensity = false;
	bool eyeCoarseCenterBigMotion = false;
	double time_start_coarse_center_cdf = 0;
	double time_end_coarse_center_cdf = 0;
	double time_start_coarse_center_color = 0;
	double time_end_coarse_center_color = 0;

	time_start_in_function = getTickCount();
	if(caculateIris_Mask_done && eyeCoarseCenterLast.y<9/16.f*FRAMEH){
		time_start_coarse_center_color = getTickCount();
		getEyeCoarseCenter = MinimalIrisColorProcess(Iris_Mask , eyeCoarseCenter , size_gaussian 
																				, IrisContour_byColor , IrisContoursPoints , getIrisContourPoints
																				, irisContour_size);	
		time_end_coarse_center_color = getTickCount();
		time_eye_position_detection_coarse_iris_location_color.push_back(time_end_coarse_center_color - time_start_coarse_center_color);
	}else{
		time_start_coarse_center_cdf = getTickCount();
		getEyeCoarseCenter = InitialCoarseCenterCDF(Gray_openingGaussian ,eyeCoarseCenter);	
		time_end_coarse_center_cdf = getTickCount();
		time_eye_position_detection_coarse_iris_location_cdf.push_back(time_end_coarse_center_cdf - time_start_coarse_center_cdf);
	}
	time_end_in_function = getTickCount();

	if(getEyeCoarseCenter){		
		getIrisContour_byIntensity = GeneralProjectionFunction_Ver2(Gray_openingGaussian , eyeCoarseCenter
																								, iris_x_regionRight , iris_x_regionLeft , iris_y_regionUp , iris_y_regionDown
																								, leftPeakDeriv_vx , rightPeakDeriv_vx
																								, vec_Deriv_GPFv_x);

		if(!getIrisContour_byIntensity){
			caculateComplete = false;
			return;
		}
	}else{
		printf("Didn't get eyeCoarseCenter!\n");
		caculateComplete = false;
		return;
	}



	if((norm(Point(eyeCoarseCenterLast.x - eyeCoarseCenter.x , eyeCoarseCenterLast.y - eyeCoarseCenter.y))>determineBigMotiondDistance)){
		eyeCoarseCenterBigMotion = true;
	}
	eyeCoarseCenterLast = eyeCoarseCenter;
	//-------------------Determine Whether EyeCoarseCenter in Extreme Region---------------------//		
	DetermineWhetherEyeCoarseCenterInExtremeRegion(eyeCoarseCenter
																					, extremeRight
																					, extremeDown
																					, extremeUp
																					, extremeLeft
																					, eyeCoarseIrisCenter_ExtremeRegion_Right
																					, eyeCoarseIrisCenter_ExtremeRegion_Down
																					, eyeCoarseIrisCenter_ExtremeRegion_Up
																					, eyeCoarseIrisCenter_ExtremeRegion_Left);


	//if(printDebug){
	//	printf("\n 3\n");
	//}
	//-------------------Eyelid Feature Detection---------------------//		
	Point start_point_upEyelid(0 , 0);
	Point start_point_downEyelid(0 , Frame.rows-1);
	vector<Point> feature_point_upperEyelid;
	vector<Point> feature_point_lowerEyelid;
	Mat Grad_X_Thresh_Pop = Mat::zeros(FRAMEH , FRAMEW , CV_8UC1);	
	Mat Grad_Y_Thresh_Pop = Mat::zeros(FRAMEH , FRAMEW , CV_8UC1);
	Mat ValleyPeakField = Mat::zeros(FRAMEH , FRAMEW , CV_8UC1);	
	Mat Gray_openingGaussian_Gau = Mat::zeros(FRAMEH , FRAMEW , CV_8UC1);	
	Mat Gray_openingGaussianOtsu = Mat::zeros(FRAMEH , FRAMEW , CV_8UC1);	
	Mat Gray_openingGaussian_Gau_inv = Mat::zeros(FRAMEH , FRAMEW , CV_8UC1);
	Point eyeRegionCenter;
	Point lineFilterCenter;
	int eyeRegionCenter_y_forEyelidDetection;

	double time_stamp1;
	double time_stamp2;
	double time_stamp3;
	double time_stamp4;
	double time_stamp5;
	//double time_stamp_1_gaussian_end = 0;
	//double time_stamp_1_core_start = 0;
	//double time_stamp_1_core_end = 0;


	//std::cout << "-----------------Gaussian begin-------------------" << std::endl;

	time_start_in_function = getTickCount();
	//cv::cuda::GpuMat dst, src1;

	//double time_upload = getTickCount();
	//src1.upload(Gray_openingGaussian);

	//double time_stamp_1_core_start = getTickCount();

	//cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createGaussianFilter(src1.type(), dst.type(), cv::Size(21, 21), 0);
	//filter->apply(src1, dst);

	//double time_stamp_1_core_end = getTickCount();

	//dst.download(Gray_openingGaussian_Gau);

	//double time_stamp_download = getTickCount();
	//imshow("GPU_GAU", Gray_openingGaussian_Gau);

	//std::cout << std::endl << "time_gpu_gau_upload = " << (time_stamp_1_core_start - time_upload) / getTickFrequency() << std::endl;
	//std::cout << std::endl << "time_gpu_gau_core = " << (time_stamp_1_core_end - time_stamp_1_core_start) / getTickFrequency() << std::endl;
	//std::cout << std::endl << "time_gpu_gau_download = " << (time_stamp_download - time_stamp_1_core_end) / getTickFrequency() << std::endl;

	//double time_cpu_gau = getTickCount();
	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_gau(Gray_openingGaussian, Gray_openingGaussian_Gau, 21, thread_num));
	
	//std::cout <<std::endl << "time_cpu_gau = " << (getTickCount() - time_cpu_gau) / getTickFrequency() << std::endl;
	//imshow("CPU_GAU", Gray_openingGaussian_Gau);
	//waitKey(0);
	//std::cout << "-----------------Gaussian end-------------------" << std::endl;
	ParallelOtsu(Gray_openingGaussian_Gau, Gray_openingGaussian_Gau_inv, cv::THRESH_BINARY_INV, thread_num);




	/*Replaced with parallel version above*/
	//GaussianBlur( Gray_openingGaussian, Gray_openingGaussian_Gau, Size(21 , 21) , 0);		
	//threshold(Gray_openingGaussian_Gau, Gray_openingGaussianOtsu, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	//Gray_openingGaussian_Gau_inv = Scalar::all(255) - Gray_openingGaussianOtsu;

	time_stamp1 = getTickCount();

	//eyeRegionCenter	
	//CenterCalculatUsingMoment(Gray_openingGaussian_Gau_inv , eyeRegionCenter.x , eyeRegionCenter.y);
	//std::cout << "OpenCV memoent = (" << eyeRegionCenter.x << ", " << eyeRegionCenter.y << ")" << std::endl;
	CenterCalculatUsingMomentParallel(Gray_openingGaussian_Gau_inv, eyeRegionCenter.x, eyeRegionCenter.y, thread_num);
	//std::cout << "CenterCalculatUsingMomentParallel memoent = (" << eyeRegionCenter.x << ", " << eyeRegionCenter.y << ")" << std::endl;
	
	//waitKey(0);


	time_stamp2 = getTickCount();

	//X Y Gradient Generation	
	//std::thread thread1(X_DirectedGradientGeneration, std::ref(Gray_openingGaussian_Gau), std::ref(Grad_X_Thresh_Pop));
	//std::thread thread2(Y_DirectedGradientGeneration, std::ref(Gray_openingGaussian_Gau), std::ref(Grad_Y_Thresh_Pop));

	//thread1.join();
	//thread2.join();
	X_DirectedGradientGeneration(Gray_openingGaussian_Gau , Grad_X_Thresh_Pop);
	Y_DirectedGradientGeneration(Gray_openingGaussian_Gau , Grad_Y_Thresh_Pop);




	time_stamp3 = getTickCount();

	//ValleyPeakField	
	EyeInnerRegionGeneration(Grad_X_Thresh_Pop , Grad_Y_Thresh_Pop , eyeRegionCenter , ValleyPeakField
											, Gray_openingGaussian_Gau_inv  
											, IrisContour_byIntensity , lineFilterCenter 
											, extremeRight , extremeLeft
											, extremeDown
											, extremeUp
											, eyeCoarseCenter												
											, leftPeakDeriv_vx , rightPeakDeriv_vx
											, vec_Deriv_GPFv_x
											, caculateIris_Mask_done , getEyeCoarseCenter , IrisContoursPoints
											, iris_y_regionUp , iris_y_regionDown
											, iris_x_regionLeft , iris_x_regionRight);	
	

	time_stamp4 = getTickCount();

	//Eyelid Detection
	if(extremeDown || extremeRight || extremeLeft || caculateIris_Mask_done){
		eyeRegionCenter_y_forEyelidDetection = eyeCoarseCenter.y;
	}else{
		eyeRegionCenter_y_forEyelidDetection = (iris_y_regionUp+iris_y_regionDown)/2.f;
	}
	time_stamp5 = getTickCount();
	
	//EyelidFeatureDetection(Gray_openingGaussian , feature_point_upperEyelid ,feature_point_lowerEyelid
	//									, eyeRegionCenter_y_forEyelidDetection , ValleyPeakField);	

	ParalleEyelidFeatureDetection(Gray_openingGaussian, feature_point_upperEyelid, feature_point_lowerEyelid
		, eyeRegionCenter_y_forEyelidDetection, ValleyPeakField, thread_num
		, time_eye_position_detection_eyelid_feature_detection_district5_core
		, time_eye_position_detection_eyelid_feature_detection_district5_merge);
	

	//ParalleEyelidFeatureDetectionLock(Gray_openingGaussian, feature_point_upperEyelid, feature_point_lowerEyelid
	//	, eyeRegionCenter_y_forEyelidDetection, ValleyPeakField, thread_num
	//	, time_eye_position_detection_eyelid_feature_detection_district5_core
	//	, time_eye_position_detection_eyelid_feature_detection_district5_merge);

	time_end_in_function = getTickCount();
	time_eye_position_detection_eyelid_feature_detection.push_back(time_end_in_function - time_start_in_function);
	time_eye_position_detection_eyelid_feature_detection_district1.push_back(time_stamp1 - time_start_in_function);
	//time_eye_position_detection_eyelid_feature_detection_district1_gaussianblur.push_back(time_stamp_1_gaussian_end - time_start_in_function);
	//time_eye_position_detection_eyelid_feature_detection_district1_gaussianblur_core.push_back(time_stamp_1_core_end - time_stamp_1_core_start);
	time_eye_position_detection_eyelid_feature_detection_district2.push_back(time_stamp2 - time_stamp1);
	time_eye_position_detection_eyelid_feature_detection_district3.push_back(time_stamp3 - time_stamp2);
	time_eye_position_detection_eyelid_feature_detection_district4.push_back(time_stamp4 - time_stamp3);
	time_eye_position_detection_eyelid_feature_detection_district5.push_back(time_end_in_function - time_stamp5);
	//-------------------Parabola Model Fitting---------------------//		
	vector<Point> upper_parabola_inlier;
	vector<Point> upper_parabola_outlier;
	vector<Point> upper_parabola_inlierSoft;
	vector<Point> lower_parabola_inlier;
	vector<Point> lower_parabola_inlierSoft;
	vector<Point> lower_parabola_outlier;
	float* lower_parabola_param;
	float* upper_parabola_param;		
	float dis_thresholdSoft_lowerEyelid;
	float dis_thresholdSoft_upperEyelid;	
	Mat EyelidRegion;
	bool getEyelidRegion;
	float *lowerParabolaTable;
	float *upperParabolaTable;
	
	time_start_in_function = getTickCount();

	//std::thread thread1(Parabola_Fitting_RANSACUp, std::ref(Frame_Gray), std::ref(feature_point_lowerEyelid), std::ref(lower_parabola_param)
	//												, std::ref(lower_parabola_inlier) , std::ref(lower_parabola_inlierSoft)
	//												, std::ref(lower_parabola_outlier) , std::ref(vertexParabolaLower) , std::ref(dis_thresholdSoft_lowerEyelid));

	//std::thread thread2(Parabola_Fitting_RANSACDown, std::ref(Frame_Gray), std::ref(feature_point_upperEyelid), std::ref(upper_parabola_param)
	//												, std::ref(upper_parabola_inlier), std::ref(upper_parabola_inlierSoft)
	//												, std::ref(upper_parabola_outlier), std::ref(vertexParabolaUpper), std::ref(dis_thresholdSoft_upperEyelid));
	//thread1.join();
	//thread2.join();
	Parabola_Fitting_RANSACUp(Frame_Gray , feature_point_lowerEyelid , lower_parabola_param
												, lower_parabola_inlier , lower_parabola_inlierSoft 
												, lower_parabola_outlier , vertexParabolaLower , dis_thresholdSoft_lowerEyelid);	

	Parabola_Fitting_RANSACDown(Frame_Gray , feature_point_upperEyelid , upper_parabola_param
												, upper_parabola_inlier, upper_parabola_inlierSoft 
												, upper_parabola_outlier, vertexParabolaUpper, dis_thresholdSoft_upperEyelid);
		

	time_end_in_function = getTickCount();

	getEyelidRegion = EyelidMaskGeneration(lower_parabola_param , upper_parabola_param
																	, EyelidRegion , vertexParabolaUpper , vertexParabolaLower
																	, lowerParabolaTable , upperParabolaTable);


	if(countForColorModelHistValidTesting==0){
		vertexParabolaUpperFirstFrame = vertexParabolaUpper;
		vertexParabolaLowerFirstFrame = vertexParabolaLower;		
	}
	vertexParabolaUpperLast = vertexParabolaUpper;
	vertexParabolaLowerLast = vertexParabolaLower;
	//if(printDebug){
	//	printf("\n 5\n");
	//}

	time_eye_position_detection_parabola_model_fitting.push_back(time_end_in_function - time_start_in_function);

	//-------------------Limbus Feature Detection---------------------//	
	bool start_point_use_coarse_pupil_detection = true;			
	vector<Point> feature_point_Limbus;	
	vector<Point> feature_point_Limbus_parallel;
	Point limbusDetection_startPoint = eyeCoarseCenter;
	Point limbusDetection_startPointParallel = eyeCoarseCenter;
	Mat Iris_Mask_forLimbus = Mat::zeros(Frame_Gray.size() , CV_8UC1);
	int final_irisRegion_Left;
	int final_irisRegion_Right;

	final_irisRegion_Left = iris_x_regionLeft;
	final_irisRegion_Right = iris_x_regionRight;

	if(!eyeCoarseCenterBigMotion){
		IrisBoundaryMedianFilter1D(iris_boundaty_Left , median_filter1DIrisBoundary_size , final_irisRegion_Left);
		IrisBoundaryMedianFilter1D(iris_boundaty_Right , median_filter1DIrisBoundary_size , final_irisRegion_Right);			
	}else{
		iris_boundaty_Left.clear();
		iris_boundaty_Right.clear();
	}

	
	IrisMaskPreProcessing_byIntensityModel(IrisContour_byIntensity , Iris_Mask_forLimbus 
																, EyelidRegion , getEyelidRegion
																, final_irisRegion_Right , final_irisRegion_Left 
																, lowerParabolaTable , upperParabolaTable
																, vertexParabolaUpper , vertexParabolaLower 
																, eyeCoarseCenter);		
	

	time_start_in_function = getTickCount();
	//LimbusFeatureDetection(Gray_openingGaussian, feature_point_Limbus, number_feature_line
	//									, limbusDetection_startPoint, start_point_use_coarse_pupil_detection , Iris_Mask_forLimbus
	//									, time_eye_position_detection_limbus_feature_detection_serial);	
	//std::cout << "(final.x, final.y) = (" << limbusDetection_startPoint.x << ", " << limbusDetection_startPoint.y << ")" << std::endl;
	//std::cout << "feature_point_Limbus.size() = " << feature_point_Limbus.size() << std::endl;

	ParalleLimbusFeatureDetection(Gray_openingGaussian, feature_point_Limbus, number_feature_line
										, limbusDetection_startPoint, start_point_use_coarse_pupil_detection, Iris_Mask_forLimbus
										, time_eye_position_detection_limbus_feature_detection_serial, thread_num);

	//std::cout << "parallel, (final.x, final.y) = (" << limbusDetection_startPointParallel.x << ", " << limbusDetection_startPointParallel.y << ")" << std::endl;
	//std::cout << "parallel, feature_point_Limbus_parallel.size() = " << feature_point_Limbus_parallel.size() << std::endl;

	//char debug_feature[MAX_WORD_LEN];
	//sprintf(debug_feature, "%s\\%s\\feature_debug.txt", testVariancePtsDir, testVarianceFolder);
	//ofstream file_time_testout(debug_feature);

	//file_time_testout << "size = " << feature_point_Limbus.size() << std::endl << "------------------" << std::endl;
	//for (int i = 0; i < feature_point_Limbus.size(); ++i) {
	//	if (i % 10 == 0) {
	//		file_time_testout << std::endl;
	//	}

	//	file_time_testout << feature_point_Limbus[i] << ", ";
	//}

	//file_time_testout.close();
	//waitKey(0);


	if(printDebug)
		printf("\n 6\n");

	time_end_in_function = getTickCount();
	time_eye_position_detection_limbus_feature_detection.push_back(time_end_in_function - time_start_in_function);

	//-------------------Refine Limbus Feature Points & IrisCenter Estimation---------------------//	
	vector<Point> clipped_feature_point_Limbus;
	vector<Point> refined_feature_point_Limbus;
	vector<Point> upperNoiseFts_Clipped;
	vector<Point> lowerNoiseFts_Clipped;
	double sum_x , sum_y;
	double mean_x , mean_y;	
	Mat FeaturePoints_LimbusFtPtsConvexHull = Mat::zeros(Frame_wh.size() , CV_8UC1);	//For extreme down
	float distLast = FLT_MAX;	
	Point tmp_pts_center = limbusDetection_startPoint;			

	time_start_in_function = getTickCount();

	//Delete those Feature Points Out of Left & Right Region	
	for(int i=0;i<feature_point_Limbus.size();++i){
		if(feature_point_Limbus[i].x>final_irisRegion_Left && feature_point_Limbus[i].x<final_irisRegion_Right){
			refined_feature_point_Limbus.push_back(feature_point_Limbus[i]);
		}
	}


	//Use All of the feature_point_Limbus to estimate eyeRefinedIrisCenter
	//Convex Hull
	vector <vector<Point>>  hull(1);			
	convexHull( Mat(refined_feature_point_Limbus), hull[0], false);			
	drawContours( FeaturePoints_LimbusFtPtsConvexHull, hull, 0, Scalar(255 , 255 , 255), cv::FILLED);//Fill the range in the convex hull	
	
	//Use Moment of Gray Level in Iris Region to Caculate Center	
	Mat MomentCenterByGrayLevel = Gray_openingGaussian.clone();
	MomentCenterByGrayLevel = Scalar::all(255) - MomentCenterByGrayLevel;
	for(int i=0;i<MomentCenterByGrayLevel.rows;++i){
		for(int j=0;j<MomentCenterByGrayLevel.cols;++j){
			if(FeaturePoints_LimbusFtPtsConvexHull.at<uchar>(i , j)!=255)
				MomentCenterByGrayLevel.at<uchar>(i , j) = 0;
		}
	}
	//std::cout << "-------------------Start------------------------" << std::endl;
	//CenterCalculatUsingMoment(MomentCenterByGrayLevel , eyeRefinedIrisCenter.x , eyeRefinedIrisCenter.y);
	//std::cout << "OpenCV memoent = (" << eyeRefinedIrisCenter.x << ", " << eyeRefinedIrisCenter.y << ")" << std::endl;

	CenterCalculatUsingMomentParallel(MomentCenterByGrayLevel, eyeRefinedIrisCenter.x, eyeRefinedIrisCenter.y, thread_num);
	//std::cout << "CenterCalculatUsingMomentParallel memoent = (" << eyeRefinedIrisCenter.x << ", " << eyeRefinedIrisCenter.y << ")" << std::endl;
	//std::cout << "-------------------End-----------------------" << std::endl;
		
	//waitKey(0);

	ClipFeaturePoints(refined_feature_point_Limbus , clipped_feature_point_Limbus
								, iris_x_regionRight , iris_x_regionLeft
								, dis_thresholdSoft_lowerEyelid , dis_thresholdSoft_upperEyelid								    
								, lower_parabola_param , upper_parabola_param
								, getEyelidRegion , EyelidRegion , upperNoiseFts_Clipped , lowerNoiseFts_Clipped);
	

	if(clipped_feature_point_Limbus.size()==0){			
		printf("No limbus feature points.\n");
		noLimbusFeaturePts = true;
		caculateComplete = false;		
		return;
	}
	
	time_end_in_function = getTickCount();
	time_eye_position_detection_center_calculation.push_back(time_end_in_function - time_start_in_function);

	//-------------------Determine Whether Iris Center has Big Motion--------------------//	
	if((norm(Point(eyeRefinedIrisCenterLast.x - eyeRefinedIrisCenter.x , eyeRefinedIrisCenterLast.y - eyeRefinedIrisCenter.y))>determineBigMotiondDistance)){
		bigmotionIrisCenter = true;
	}


	eyeRefinedIrisCenterLast = eyeRefinedIrisCenter;
	//-------------------Refresh Extreme Region Limit of EyeCoarseCenter--------------------//	
	RefreshExtremeRegionLimit_eyeCoarseCenter(eyeRefinedIrisCenter 
																			, eyeCoarseIrisCenter_ExtremeRegion_Right
																			, eyeCoarseIrisCenter_ExtremeRegion_Down
																			, eyeCoarseIrisCenter_ExtremeRegion_Up
																			, eyeCoarseIrisCenter_ExtremeRegion_Left
																			, eyeCenter_rightestPos
																			, eyeCenter_lowestPos
																			, eyeCenter_uppestPos
																			, eyeCenter_leftestPos);

	//-------------------Determine Whether Iris Center in Extreme Region--------------------//	
	DetermineWhetherEyeRefineCenterInExtremeRegion(eyeRefinedIrisCenter
																					, extremeRight_forBlink
																					, extremeDown_forBlink
																					, extremeUp_forBlink
																					, extremeLeft_forBlink
																					, eyeCoarseIrisCenter_ExtremeRegion_Right
																					, eyeCoarseIrisCenter_ExtremeRegion_Down
																					, eyeCoarseIrisCenter_ExtremeRegion_Up
																					, eyeCoarseIrisCenter_ExtremeRegion_Left);




	
	//if(printDebug){
	//	printf("\n 7\n");
	//}
	//-------------------Determine Whether Refresh the Iris Histogram Model--------------------//	
	float max_contourSize = -FLT_MAX;
	int maxContourID;

	if(!calibrationProcedureBegin){
		if(frame_number%iris_maskModel_refreshFrame==0){				
			if(!bigmotionIrisCenter && getEyelidRegion){			
				if(EyelidRegion.at<uchar>(eyeRefinedIrisCenter.y , eyeRefinedIrisCenter.x)==255){				
					//Get Rotated Rect of minArea of Object(Convex Hill of Iris)		
					time_start_in_function = getTickCount();

					Mat testROI = Frame_wh.clone();
					Mat Refined_LimnusFtPtsConvexHull_Processed = Mat::zeros(Frame_wh.size() , CV_8UC1);	//For extreme down;
					vector<vector<Point> > contours;
					vector<Vec4i> hierarchy;		

					
					float morphological_erode_refinedEyeConvexHull_size_Cal = countNonZero(FeaturePoints_LimbusFtPtsConvexHull)/eyeRefinedConvexForIrisMaskColorModelBaseSize_experiment*morphological_erode_refinedEyeConvexHull_size;
					Mat morph_ellipse_element = getStructuringElement(MORPH_ELLIPSE, Size(2 * morphological_erode_refinedEyeConvexHull_size_Cal + 1, 2 * morphological_erode_refinedEyeConvexHull_size_Cal + 1), Point(morphological_erode_refinedEyeConvexHull_size_Cal, morphological_erode_refinedEyeConvexHull_size_Cal));
					/*Morphology_Operations(FeaturePoints_LimbusFtPtsConvexHull, Refined_LimnusFtPtsConvexHull_Processed , MORPH_ERODE
														, morphological_erode_refinedEyeConvexHull_size_Cal,  MORPH_ELLIPSE);*/	

					cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_er(FeaturePoints_LimbusFtPtsConvexHull
												, Refined_LimnusFtPtsConvexHull_Processed , morph_ellipse_element, thread_num));

						
					//Mat Refined_LimnusFtPtsConvexHull_find = Refined_LimnusFtPtsConvexHull_Processed.clone();					
					findContours( Refined_LimnusFtPtsConvexHull_Processed, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, Point(0, 0) );
					
					if(contours.size()!=0){
						for(int i=0;i<contours.size();++i){		
							if(contours[i].size()>max_contourSize){
								max_contourSize = contours[i].size();
								maxContourID = i;
							}
						}
					}							
					if(contours.size()==0 || max_contourSize<50){						
						morphological_erode_refinedEyeConvexHull_size -= 2;				
					}else{								
						RotatedRect minRect = minAreaRect(Mat(contours[maxContourID]));
						Point2f rect_points[4]; 
						minRect.points( rect_points );
						for( int j = 0; j < 4; j++ )
							line( testROI, rect_points[j], rect_points[(j+1)%4], Scalar(0 , 255 , 255), 3, 8 );
				
						//Get ROI to Generate Hist Model
						Mat rotatedMatrix, rotated;
						// get angle and size from the bounding box
						float angle = minRect.angle;
						Size rect_size = minRect.size;
					
						if (minRect.angle < -45.0) {
							angle += 90.0;
							swap(rect_size.width, rect_size.height);
						}				
						// get the rotation matrix						
						rotatedMatrix = getRotationMatrix2D(minRect.center, angle, 1.0);
						// perform the affine transformation						
						warpAffine(Frame_wh, rotated, rotatedMatrix ,  Frame_wh.size(), INTER_CUBIC);
						// crop the resulting image	

						double time_start1 = getTickCount();
						Iris_ROI_forModel = Mat::zeros(rect_size , CV_8U);
						getRectSubPix(rotated, rect_size, minRect.center, Iris_ROI_forModel);
						time_eye_position_detection_refresh_iris_ROI_region.push_back(getTickCount() - time_start1);
											
						gotIrisROI = true;						
					}

					time_end_in_function = getTickCount();
					time_eye_position_detection_refresh_min_area_region.push_back(time_end_in_function - time_start_in_function);

					//std::cout << "getIrisROI = " << gotIrisROI << std::endl;
					if (!(contours.size() == 0 || max_contourSize < 50)) {
						imshow("testROI", testROI);
						imshow("Iris_ROI_forModel", Iris_ROI_forModel);
						writer_result_roi.write(testROI);
					}
				}
			}
		}	
	}

	//if(printDebug){
	//	printf("\n 8\n");
	//}
	//-------------------Get EyelidRegion & IrisRegion of Sample Image for ColorHistModel Testing--------------------//	
	if(countForColorModelHistValidTesting==0){
		if(eyeRefinedIrisCenter.x>FRAMEW/2){
			eyeCenter_rightestPos = eyeRefinedIrisCenter.x;
		}else{
			eyeCenter_rightestPos = (eyeRefinedIrisCenter.x+rightCornerOriginalPoint.x)/2;
		}

		IrisRegionValidTesting = Mat::zeros(Frame_Gray.size() , CV_8UC1);				

		//Erosion	
		float morphological_erode_refinedEyeConvexHull_size_Cal = countNonZero(FeaturePoints_LimbusFtPtsConvexHull)/eyeRefinedConvexForIrisMaskColorModelBaseSize_experiment*morphological_erode_IrisRegionValidTesting_size;
		Mat morph_ellipse_element = getStructuringElement(MORPH_ELLIPSE, Size(2 * morphological_erode_refinedEyeConvexHull_size_Cal + 1, 2 * morphological_erode_refinedEyeConvexHull_size_Cal + 1), Point(morphological_erode_refinedEyeConvexHull_size_Cal, morphological_erode_refinedEyeConvexHull_size_Cal));
		/*Morphology_Operations(FeaturePoints_LimbusFtPtsConvexHull, IrisRegionValidTesting , MORPH_ERODE
											, morphological_erode_refinedEyeConvexHull_size_Cal,  MORPH_ELLIPSE);	*/
		cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_er(FeaturePoints_LimbusFtPtsConvexHull
												, IrisRegionValidTesting , morph_ellipse_element, thread_num));
		eyeCloseDetermine_irisContourSizeThreshold_colorModelBased = countNonZero(IrisRegionValidTesting)*eyeCloseDetermine_irisContourSizeThreshold_percent;				
	}

	//if(printDebug){
	//	printf("\n 9\n");
	//}

	//------------------Display Result-----------------//	
	//Display Iris Region			
	for(int i=0;i<EyePosition_Result.rows;++i){
		for(int j=0;j<EyePosition_Result.cols;++j){
			if(FeaturePoints_LimbusFtPtsConvexHull.at<uchar>(i , j)==255){
				EyePosition_Result.at<Vec3b>(i , j) = Vec3b(255 , 255 , 255);
			}
		}
	}
	
	DrawPrabolaVer3(EyePosition_Result  , upperParabolaTable , 50 , getEyelidRegion);
	DrawPrabolaVer2(EyePosition_Result  , lowerParabolaTable , EyePosition_Result , 50 , getEyelidRegion);	

	circle(EyePosition_Result , eyeRefinedIrisCenter , 20 , Scalar(0 , 100 , 255) , 2);

	line(EyePosition_Result , Point(0 , eyeRefinedIrisCenter.y) , Point(FRAMEW-1 , eyeRefinedIrisCenter.y)
		,Scalar(0 , 255 , 255) , 3);
	line(EyePosition_Result , Point(eyeRefinedIrisCenter.x , 0 ) , Point(eyeRefinedIrisCenter.x , FRAMEH-1)
		, Scalar(0 , 255 , 255) , 3);	

	Draw_Cross(EyePosition_CenterResult , eyeRefinedIrisCenter.x , eyeRefinedIrisCenter.y , 35 , 35 , Scalar(0 , 20 , 255) , 3);	
	
	//if(printDebug){
	//	printf("\n 10\n");
	//}
	
	upper_parabola_inlier.clear();
	upper_parabola_outlier.clear();
	lower_parabola_inlier.clear();
	lower_parabola_inlierSoft.clear();
	lower_parabola_outlier.clear();	
	feature_point_Limbus.clear();
	feature_point_upperEyelid.clear();
	feature_point_lowerEyelid.clear();		

	delete [] lower_parabola_param;	
	delete [] upper_parabola_param;	
	//if(printDebug){
	//	printf("\n 11\n");
	//}

	caculateComplete = true;
	return;
}


inline void ReadGroundTruth(vector<Point> &groundTruth , char inputFileName[]){
	ifstream f_in(inputFileName);

	char temp[MAX_WORD_LEN];
	while(f_in>>temp){
		float x;		
        istringstream iss_fir(temp);        
        iss_fir>>dec>>x;
		float y; 	
		f_in>>temp;
		istringstream iss_sec(temp);
		iss_sec>>dec>>y;
		groundTruth.push_back(Point(x , y));
	}
}



inline void ScleraModelGeneration(const Mat &ScleraROI , Mat &Sclera_hist, int channels[] , int histSize[] , const float *ranges[]){    
	Mat ScleraROI_HSV;

	cvtColor(ScleraROI, ScleraROI_HSV, COLOR_BGR2HSV);
	calcHist( &ScleraROI_HSV, 1, channels, Mat(), // do not use mask
				 Sclera_hist, 3, histSize, ranges,
				 true, // the histogram is uniform
				 false );
}

inline void IrisModelGeneration(const Mat &IrisROI , Mat &Iris_hist , int channels[] , int histSize[] , const float *ranges[]){    
	Mat IrisROI_HSV;

	double time_start = getTickCount();

	cvtColor(IrisROI, IrisROI_HSV, COLOR_BGR2HSV);
	calcHist( &IrisROI_HSV, 1, channels, Mat(), // do not use mask
				 Iris_hist, 2, histSize, ranges,
				 true, // the histogram is uniform
				 false );

	time_iris_model_calculate_h_s_histogram.push_back(getTickCount() - time_start);

	//-------------------Show Histogram Info------------------------//
	//std::cout << "IrisROI.rows = " << IrisROI.rows << std::endl;
	//std::cout << "IrisROI.cols = " << IrisROI.cols << std::endl;

	//for (int i = 0; i < Iris_hist.rows; ++i) {
	//	std::cout << std::endl;
	//	for (int j = 0; j < Iris_hist.cols; ++j) {
	//		std::cout << Iris_hist.at<float>(i, j) <<" ";
	//	}
	//}

	//int scale = 100;
	//double maxVal = 0;
	//minMaxLoc(Iris_hist, 0, &maxVal, 0, 0);
	//Mat histImg = Mat::zeros(Iris_hist.cols*scale, Iris_hist.rows * scale, CV_8UC3);
	//for (int h = 0; h < Iris_hist.rows; h++)
	//	for (int s = 0; s < Iris_hist.cols; s++)
	//	{
	//		float binVal = Iris_hist.at<float>(h, s);
	//		int intensity = cvRound(binVal * 255 / maxVal);
	//		rectangle(histImg, Point(h*scale, s*scale),
	//			Point((h + 1)*scale - 1, (s + 1)*scale - 1),
	//			Scalar::all(intensity),
	//			CV_FILLED);
	//	}

	//namedWindow("H-S Histogram", 1);
	//imshow("H-S Histogram", histImg);
	//waitKey(0);

}

inline void CalcBackProjSelfDefined(const Mat &Src , const MatND &HistModel , Mat &outMask , const int histSize[] 
													, const int Hist_PerBin[]){														
	outMask = Mat::zeros(Src.rows , Src.cols , CV_8UC1);
	Mat Src_HSV;

	cvtColor(Src , Src_HSV , COLOR_BGR2HSV);

	double maxVal = -DBL_MAX;	
	for(int i=0;i<histSize[0];++i){
		for(int j=0;j<histSize[1];++j){		
			if(maxVal<HistModel.at<float>(i , j))
					maxVal=HistModel.at<float>(i , j);			
		}
	}

	for(int i=0;i<Src_HSV.rows;++i){
		for(int j=0;j<Src_HSV.cols;++j){
			int pos_x = Src_HSV.at<Vec3b>(i , j)[1]/(float)Hist_PerBin[1];//S
			int pos_y = Src_HSV.at<Vec3b>(i , j)[0]/(float)Hist_PerBin[0];//H			
			outMask.at<uchar>(i , j) = HistModel.at<float>(pos_y , pos_x)/maxVal*255;				
		}
	}
}

inline cv::Mat colorReduce(const cv::Mat &image, int div=64) {

	  int n= static_cast<int>(log(static_cast<double>(div))/log(2.0));
	  // mask used to round the pixel value
	  uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0

	  cv::Mat_<cv::Vec3b>::const_iterator it= image.begin<cv::Vec3b>();
	  cv::Mat_<cv::Vec3b>::const_iterator itend= image.end<cv::Vec3b>();

	  // Set output image (always 1-channel)
	  cv::Mat result(image.rows,image.cols,image.type());
	  cv::Mat_<cv::Vec3b>::iterator itr= result.begin<cv::Vec3b>();

	  for ( ; it!= itend; ++it, ++itr) {
        
        (*itr)[0]= ((*it)[0]&mask) + div/2;
        (*itr)[1]= ((*it)[1]&mask) + div/2;
        (*itr)[2]= ((*it)[2]&mask) + div/2;
	  }

	  return result;
}

inline void ReadScleraModel(bool &readScleraModel , Mat &Sclera_hist , bool &scleraDynamicMaskGeneration
							 , char ScleraHisto_output_fileName[]){	
	int dim1 , dim2 , dim3;	
	ifstream fin(ScleraHisto_output_fileName);
	string in_fileString;
	vector<float> file_input_val;
	int count = 0;

	while(fin>>in_fileString){
		istringstream  iss(in_fileString);
		float input_val;
		iss>>dec>>input_val;
		if(count == 0){
			dim1 = input_val;
		}else if(count==1){
			dim2 = input_val;
		}else if(count==2){
			dim3 = input_val;
		}else{
			file_input_val.push_back(input_val);
		}	
		++count;
	}								
	const int mySizes[3]={dim1 , dim2 , dim3};
				
	Mat readSclera_model(3 , mySizes , CV_32FC1 ,Scalar(0))	;

	for(int i=0;i<dim1;++i){
		for(int j=0;j<dim2;++j){
			for(int k=0;k<dim3;++k){
				readSclera_model.at<float>(i , j , k) = file_input_val[i*dim2*dim3+j*dim3+k];
			}
		}
	}
	Sclera_hist = readSclera_model.clone();
	readScleraModel = false;
	scleraDynamicMaskGeneration = true;
}

inline void ReadIrisModel(Mat &Iris_hist , char IrisHisto_output_fileName[]){	
	int dim1 , dim2;
	ifstream fin(IrisHisto_output_fileName);
	string in_fileString;
	vector<float> file_input_val;
	int count = 0;

	while(fin>>in_fileString){
		istringstream  iss(in_fileString);
		float input_val;
		iss>>dec>>input_val;
		if(count == 0){
			dim1 = input_val;
		}else if(count==1){
			dim2 = input_val;
		}else{
			file_input_val.push_back(input_val);
		}	
		++count;
	}								
	const int mySizes[2]={dim1 , dim2};
				
	Mat readIris_model(2 , mySizes , CV_32FC1 ,Scalar(0))	;

	for(int i=0;i<dim1;++i){
		for(int j=0;j<dim2;++j){			
			readIris_model.at<float>(i , j) = file_input_val[i*dim1+j];
		}
	}
	Iris_hist = readIris_model.clone();
}


inline bool HistoModelValidTesting(const Mat &EyeImageForTestingIrisHistModel , const Mat &Iris_hist 
													, const Mat &IrisRegionValidTesting
													, int histSize[] , const int Iris_PerBin[])
{	
	Mat Iris_Mask;
	Mat Iris_Mask_Gau = Mat::zeros(EyeImageForTestingIrisHistModel.size() , CV_8UC1);
	Mat Iris_Mask_Otsu = Mat::zeros(EyeImageForTestingIrisHistModel.size() , CV_8UC1);
	float irisRegion_size = countNonZero(IrisRegionValidTesting);
	float irisRate;
	bool validHistModel = false;
	int pixel_in_iris = 0;
	int pixel_in_others = 1;
	const int size_gaussian = 61;

	double time_start = getTickCount();
	//Use the Iris_hist Model to Cauclate Iris Mask for Testing
	CalcBackProjSelfDefined(EyeImageForTestingIrisHistModel , Iris_hist , Iris_Mask , histSize , Iris_PerBin);	
	//GaussianBlur( Iris_Mask, Iris_Mask_Gau, Size(size_gaussian,size_gaussian) , 0);			
	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_gau(Iris_Mask, Iris_Mask_Gau, size_gaussian, thread_num));
	//threshold(Iris_Mask_Gau, Iris_Mask_Otsu, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	ParallelOtsu(Iris_Mask_Gau, Iris_Mask_Otsu, cv::THRESH_BINARY, thread_num);

	//Caculate the Iris rate = pixel_in_iris/pixil_int_others
	//for(int i=0;i<Iris_Mask_Otsu.rows;++i){
	//	for(int j=0;j<Iris_Mask_Otsu.cols;++j){
	//		if(Iris_Mask_Otsu.at<uchar>(i , j)==255){
	//			if(IrisRegionValidTesting.at<uchar>(i , j) == 255){
	//				++pixel_in_iris;
	//			}else{
	//				++pixel_in_others;
	//			}
	//		}
	//	}
	//}
	//irisRate = pixel_in_iris/(float)pixel_in_others;

	//std::cout << std::endl << "serial, pixel_in_iris = " << pixel_in_iris << std::endl;
	//std::cout << std::endl << "serial, pixel_in_others = " << pixel_in_others << std::endl;
	//std::cout << std::endl << "serial, irisRate = " << irisRate << std::endl;

	ParallelCalcIrisRate(Iris_Mask_Otsu, IrisRegionValidTesting, irisRate, pixel_in_iris, pixel_in_others, thread_num);
	//std::cout << std::endl << "parallel, irisRate = " << irisRate << std::endl;
	//waitKey(0);

	time_iris_model_valid_testing.push_back(getTickCount() - time_start);

	if(irisRate>irisRate_max && irisRate>iris_colorModelValidTestingIrisRate_initial){
		if(pixel_in_others==1){			
			if(irisRate>iris_colorModelIrisRate_pixelInOthersOne*irisRegion_size){				
				irisRate_max = irisRate;
				validHistModel = true;
/*				imshow("IrisRegionValidTesting" , IrisRegionValidTesting);
				imshow("Iris_Mask_OtsuInValidTesting" , Iris_Mask_Otsu);
				imshow("Iris_MaskInValidTesting" , Iris_Mask);
				imshow("EyeImageForTestingIrisHistModel" , EyeImageForTestingIrisHistModel);*/						
			}else{
				validHistModel = false;
			}
		}else{			
			irisRate_max = irisRate;
			validHistModel = true;
/*			imshow("IrisRegionValidTesting" , IrisRegionValidTesting);
			imshow("Iris_Mask_OtsuInValidTesting" , Iris_Mask_Otsu);
			imshow("Iris_MaskInValidTesting" , Iris_Mask);
			imshow("EyeImageForTestingIrisHistModel" , EyeImageForTestingIrisHistModel);	*/			
		}
	}else{		
		validHistModel = false;
	}

	return validHistModel;
}


inline void IrisModelHandeling(bool &gotIrisROI , bool &irisDynamicMaskGeneration , bool &readIrisModel
								  , bool &writeIrisModel , char IrisHisto_output_fileName[] 
								  , const Mat &Iris_ROI_forModel  , const Mat &Frame_wh , Mat &Iris_hist , Mat &Iris_Mask
								  , int channels[] , int histSize[] , const float *ranges[] , const int Iris_PerBin[] , vector<Mat> &Iris_hist_vector
								  , const Mat &EyeImageForTestingIrisHistModel
								  , const Mat &IrisRegionValidTesting
								  , bool &caculateIris_Mask_done , int &countRefreshTimes)
{
	bool irisHist_IsValidModel = false;
	bool checkIrisColorModel = false;
	Mat Iris_hist_cal;
	if(gotIrisROI){		
		IrisModelGeneration(Iris_ROI_forModel , Iris_hist_cal ,channels , histSize , ranges);		
		gotIrisROI = false;		
		readIrisModel = false;
		checkIrisColorModel = true;
	}

	if(readIrisModel){		
		ReadIrisModel(Iris_hist , IrisHisto_output_fileName);
		Iris_hist_vector.push_back(Iris_hist);
		readIrisModel = false;
		irisDynamicMaskGeneration = true;
	}

	if(checkIrisColorModel){	
		irisHist_IsValidModel = HistoModelValidTesting(EyeImageForTestingIrisHistModel , Iris_hist_cal
																			, IrisRegionValidTesting , histSize , Iris_PerBin);
		
		if(irisHist_IsValidModel){
			Iris_hist = Iris_hist_cal.clone();
			irisDynamicMaskGeneration = true;
		}
		checkIrisColorModel = false;
	}

	if(irisDynamicMaskGeneration){				
		CalcBackProjSelfDefined(Frame_wh , Iris_hist , Iris_Mask , histSize , Iris_PerBin);	
		if(countRefreshTimes==0){
			++countRefreshTimes;
		}
		caculateIris_Mask_done = true;			
	}


	if(writeIrisModel){		
		ofstream fout(IrisHisto_output_fileName);						
		fout<<histSize[0]<<" "<<histSize[1]<<endl;
		fout<<endl;
		for(int i=0;i<histSize[0];++i){
			for(int j=0;j<histSize[1];++j){												
				fout<<Iris_hist.at<float>(i , j)<<" ";
			}
			fout<<endl;
		}
		writeIrisModel = false;
	}

}




inline void WriteOutTestingData(const char *testVarianceWriteOutFileDir){
	//char centerEstimation_coarseCenter_fileName[MAX_WORD_LEN];
	//char centerEstimation_ellipseFineCenter_fileName[MAX_WORD_LEN];
	char centerEstimation_convexHullFineCenter_fileName[MAX_WORD_LEN];
	//char ellipseArea_fileName[MAX_WORD_LEN];
	//char convexHullArea_fileName[MAX_WORD_LEN];

	//sprintf(centerEstimation_coarseCenter_fileName , "%s\\centerEstimation_coarseCenter.txt" , testVarianceWriteOutFileDir);
	//sprintf(centerEstimation_ellipseFineCenter_fileName , "%s\\centerEstimation_ellipseFineCenter.txt" , testVarianceWriteOutFileDir);
	sprintf(centerEstimation_convexHullFineCenter_fileName , "%s\\centerEstimation_convexHullFineCenter_vfinal.txt" , testVarianceWriteOutFileDir);
	//sprintf(ellipseArea_fileName , "%s\\ellipseArea.txt" , testVarianceWriteOutFileDir);
	//sprintf(convexHullArea_fileName , "%s\\convexHullArea.txt" , testVarianceWriteOutFileDir);


	//ofstream fout_centerEstimation_coarseCenter(centerEstimation_coarseCenter_fileName);
	//ofstream fout_centerEstimation_ellipseFineCenter(centerEstimation_ellipseFineCenter_fileName);
	ofstream fout_centerEstimation_convexHullFineCenter(centerEstimation_convexHullFineCenter_fileName);
	//ofstream fout_ellipseArea(ellipseArea_fileName);
	//ofstream fout_convexHullArea(convexHullArea_fileName);

	//for(int i=0;i<centerEstimation_coarseCenter.size();++i){
	//	fout_centerEstimation_coarseCenter<<centerEstimation_coarseCenter[i].x<<" "<<centerEstimation_coarseCenter[i].y<<endl;
	//}
	//for(int i=0;i<centerEstimation_ellipseFineCenter.size();++i){
	//	fout_centerEstimation_ellipseFineCenter<<centerEstimation_ellipseFineCenter[i].x<<" "<<centerEstimation_ellipseFineCenter[i].y<<endl;
	//}
	for(int i=0;i<centerEstimation_convexHullFineCenter.size();++i){
		fout_centerEstimation_convexHullFineCenter<<centerEstimation_convexHullFineCenter[i].x<<" "<<centerEstimation_convexHullFineCenter[i].y<<endl;
	}
	//for(int i=0;i<ellipseArea.size();++i){
	//	fout_ellipseArea<<ellipseArea[i]<<endl;
	//}
	//for(int i=0;i<convexHullArea.size();++i){
	//	fout_convexHullArea<<convexHullArea[i]<<endl;
	//}
}

inline void WriteOutEyeCenterGroundTruthData(const char *OutFileName){
	ofstream fout_groundTruth(OutFileName);

	for(int i=0;i<groundTruthWriteOut.size();++i){
		fout_groundTruth<<groundTruthWriteOut[i].x<<" "<<groundTruthWriteOut[i].y<<endl;
	}
}

#pragma endregion

inline bool CalibrationProcedure(Mat &Scene_cal , int &count_times_2secs 
												, int &posLine_y , 	const int &x_dist , const int &y_dist
												, Point &calibratedPts
												, const bool &posLineDivisible2 , const int &calBrationMethod
												, const bool &y_inFirstORLastRows)											
{
	int x_initial = x_dist/2;
	int y_initial = y_dist/2;
	int posX , posY;
	int posX_interA , posY_interA;
		
	posY = y_initial+posLine_y*y_dist;
	posY_interA = posY;

	if(calBrationMethod==calibrationMethod::HomographySliceMapping){				
		posX = x_initial + count_times_2secs*x_dist;
	}else{
		if(posLineDivisible2){
			posX = x_initial + (count_times_2secs-2)*x_dist;
			posX_interA = x_initial + (count_times_2secs-1)*x_dist;
		}else{		
			posX = x_initial + (count_times_2secs-1)*x_dist;
			posX_interA = x_initial + (count_times_2secs-2)*x_dist;
		}
	}	

	circle(Scene_cal , Point(posX , posY) , MIN(x_dist , y_dist)/2-5 , Scalar(150 , 200 , 50) , cv::FILLED);
	Draw_Cross(Scene_cal , posX , posY , MIN(x_dist , y_dist)/10 , MIN(x_dist , y_dist)/10 , Scalar(0 , 10 , 255) , 2);
	
	calibratedPts.x = posX;
	calibratedPts.y = posY;
	
	return true;
}


inline void DrawChessBoard(Mat &Scene_calibration , const int &x_dist , const int &y_dist 
											, vector<Point> &calibrationPatternPtsStep_One
											, const int &numXSpace , const int &numYSpace){		
	if(calibrationPts_space==calibrationPattern::Step_space_two){
		int half_y_dist = y_dist/2;
		int half_x_dist = x_dist/2;
		for(int i=0;i<Scene_calibration.rows;++i){
			if(i%y_dist==0){
				line(Scene_calibration , Point(0 , i) , Point(Scene_calibration.cols - 1 , i) , Scalar(0 , 150 , 255) , 2);			
			}
		}
		for(int j=0;j<Scene_calibration.cols;++j){
			if(j%x_dist==0){
				line(Scene_calibration , Point(j , 0) , Point(j , Scene_calibration.rows - 1) , Scalar(0 , 150 , 255) , 2);		
			}
		}	
	}else{
		int half_y_dist = y_dist/2;
		int half_x_dist = x_dist/2;
		for(int i=0;i<numYSpace;++i){
			for(int j=0;j<numXSpace;++j){				
				Point tmp(half_x_dist+j*x_dist , half_y_dist+i*y_dist);
				calibrationPatternPtsStep_One.push_back(tmp);						
			}
		}		
		for(int i=1;i<numYSpace;++i){
			line(Scene_calibration , Point(0 , i*y_dist) , Point(Scene_calibration.cols - 1 , i*y_dist) , Scalar(0 , 150 , 255) , 2);	
		}
		for(int i=1;i<numXSpace;++i){
			line(Scene_calibration , Point(i*x_dist , 0) , Point(i*x_dist , Scene_calibration.rows - 1) , Scalar(0 , 150 , 255) , 2);		
		}
	}
}

inline void AnchorPointsCollenction(const int &count_line_cols , const int &count_line_rows													
													, vector<Point> &calInterAnchorPoints)
{
	for(int i=0;i<count_line_rows;++i){
		for(int j=0;j<count_line_cols;++j){
			if(i%2==0){
				if(j%2!=0){
					calInterAnchorPoints.push_back(Point(j , i));
				}
			}else{
				if(j%2==0){
					calInterAnchorPoints.push_back(Point(j , i));
				}
			}
		}
	}
}

inline bool CalibrationProcedureWhole(Mat &Scene_chessboard , Mat &Scene_calibration , int &count_calProcedure , double &time_cal
														, int &time_duringLast , int &count_times_2secs , int &posLine_y
														, const int &x_dist , const int &y_dist 
														, bool &getCalibrationPoints , const bool &setEyeCornerAndEyePosReady
														, int &test_count_cal , bool &calibrationProcedureBegin
														, const Point &eyeRefinedIrisCenter 
														, vector<Point> &calibratedEyeRefinedCenter , vector<Point> &calibratedCalPoints
														, deque<int> &posQueueX , deque<int> &posQueueY
														, Point &filteredEyeRefinedCenter
														, const bool &eyeState														
														, Point &calibratedChessBoardPtsLast
														, vector<Point> &calInterAnchorPoints
														, const int &calBrationMethod
														, int &count_times_2secsNext
														, bool &isChessBoardSideOdd
														, const int &numXSpace , const int &numYSpace
														, int &countEyePut														
														, vector<Point> &calibrationPatternPtsStep_One
														, int &countCalibrationPatternPtsStep_One_pos)
{
	bool calibration_CollectDataDone = false;
	Point calibratedChessBoardPts = calibratedChessBoardPtsLast;	
	if(count_calProcedure==0){
		//Delete old files
		char tmpOutFileName[MAX_WORD_LEN];
		sprintf(tmpOutFileName , "del -f -q %s\\*.jpg" , analysisGazeOutputDir);
		system(tmpOutFileName);
		sprintf(tmpOutFileName , "del -f -q %s\\*.txt" , analysisGazeOutputDir);
		system(tmpOutFileName);

		calibratedEyeRefinedCenter.clear();
		calibratedCalPoints.clear();
		calInterAnchorPoints.clear();
		calibrationPatternPtsStep_One.clear();
		DrawChessBoard(Scene_chessboard , x_dist , y_dist , calibrationPatternPtsStep_One
									, numXSpace , numYSpace);			
		isChessBoardSideOdd = (numXSpace%2==0)?false:true;
		if(calBrationMethod==calibrationMethod::HomographySliceMapping){					
			if(!isChessBoardSideOdd){
				printf("\nIn HomographySliceMapping method , callibration points line must be odd.\n");
				return false;
			}
		}

		Scene_calibration = Scene_chessboard.clone();
		imshow("Scene" , Scene_calibration);				
		waitKey(0);
		time_cal = getTickCount(); //start to calculate time 				
	}

	int time_during = (getTickCount() - time_cal)/getTickFrequency();						
	if((time_during%calibrationInterTimePoints==0 && time_duringLast!=time_during) || count_calProcedure==0){				
		bool posLineDivisible2 = (posLine_y%2==0)?true:false;
		Point interpolationAnchorPts;
		bool y_inFirstORLastRows = false;		
		getCalibrationPoints = true;

		//Collect filteredEyeRefinedCenter in the past time
		if(count_calProcedure!=0 && (eyeState == Eyeopen )	&& setEyeCornerAndEyePosReady){
//#pragma region						
//			char test_imageFille[MAX_WORD_LEN];
//			Mat TestCalibrationFilteredEye = Frame_wh.clone();
//			Draw_Cross(TestCalibrationFilteredEye , filteredEyeRefinedCenter.x , filteredEyeRefinedCenter.y , 30 , 30 , Scalar(0 , 255 , 50) , 2);	
//			sprintf(test_imageFille , "%s\\Eye_image_%d.jpg" , analysisGazeOutputDir , time_during);
//			imwrite(test_imageFille, TestCalibrationFilteredEye);						
//#pragma endregion

			calibratedEyeRefinedCenter.push_back(filteredEyeRefinedCenter);
		}



		//Calibration Pattern Choosing
		Scene_calibration = Scene_chessboard.clone();
		if(calibrationPts_space==calibrationPattern::Step_space_two){
			if(calBrationMethod==calibrationMethod::HomographySliceMapping){
				if(posLine_y==0 || posLine_y==numYSpace-1){
					y_inFirstORLastRows = true;
					count_times_2secs = count_times_2secsNext;
					count_times_2secsNext = count_times_2secs+1;	
					if(posLine_y==0){
						if(count_times_2secs>numXSpace-1){
							count_times_2secs = 0;
							count_times_2secsNext = 1;
							++posLine_y;
						}
					}else{
						if(count_times_2secs>numXSpace-1){
							count_times_2secs = 0;
							count_times_2secsNext = 0;
							++posLine_y;
						}
					}
				}else{
					if(posLineDivisible2){
						count_times_2secs+=2;
						if(count_times_2secs>numXSpace-1){
							count_times_2secs = 0;
							++posLine_y;
						}
					}else{
						count_times_2secs = count_times_2secsNext;
						if(count_times_2secs==0){
							count_times_2secsNext = 1;
						}else if(count_times_2secs==numXSpace-2){
							count_times_2secsNext = numXSpace-1;
						}else if(count_times_2secs==numXSpace-1){
							count_times_2secsNext = count_times_2secs+1;					
						}else{
							count_times_2secsNext = count_times_2secs+2;
						}	

						if(count_times_2secsNext>numXSpace){
							count_times_2secs = 0;
							count_times_2secsNext = 1;
							++posLine_y;
						}
					}
				}		
			}else{
				count_times_2secs+=2;
				if(posLineDivisible2){
					if(count_times_2secs>numXSpace+1){
						count_times_2secs = 2;
						++posLine_y;
					}
				}else{
					if(count_times_2secs>numXSpace){
						count_times_2secs = 2;
						++posLine_y;
					}
				}
			}
			posLineDivisible2 = (posLine_y%2==0)?true:false;


			//Display Calibration Points							
			if(posLine_y>numYSpace-1){
				calibration_CollectDataDone = true;
				posLine_y = 0;
				count_times_2secs = 0;		
				getCalibrationPoints = false;
			}else{					
				CalibrationProcedure(Scene_calibration , count_times_2secs 
												, posLine_y , x_dist , y_dist , calibratedChessBoardPts 
												, posLineDivisible2 , calBrationMethod
												, y_inFirstORLastRows);	
				countEyePut = 0;			
			}
		}else{//calibrationPattern::Step_space_one
			if(countCalibrationPatternPtsStep_One_pos>calibrationPatternPtsStep_One.size()-1){
				calibration_CollectDataDone = true;
				countCalibrationPatternPtsStep_One_pos = 0;
				count_times_2secs = 0;		
				getCalibrationPoints = false;
			}else{
				circle(Scene_calibration , calibrationPatternPtsStep_One[countCalibrationPatternPtsStep_One_pos] , MIN(x_dist , y_dist)/2-5 , Scalar(150 , 200 , 50) , cv::FILLED);
				Draw_Cross(Scene_calibration , calibrationPatternPtsStep_One[countCalibrationPatternPtsStep_One_pos].x 
									, calibrationPatternPtsStep_One[countCalibrationPatternPtsStep_One_pos].y 
									, MIN(x_dist , y_dist)/10 , MIN(x_dist , y_dist)/10 , Scalar(0 , 10 , 255) , 2);

				calibratedChessBoardPts = calibrationPatternPtsStep_One[countCalibrationPatternPtsStep_One_pos];
				++countCalibrationPatternPtsStep_One_pos;			
			}
		}
		//Collect Eye Position When Calibration Points Displayed
		if(getCalibrationPoints && setEyeCornerAndEyePosReady){
			posQueueX.clear();
			posQueueY.clear();
			calibratedCalPoints.push_back(calibratedChessBoardPts);			
		}	


	}//End timduring Pts Moving




	//Median Filter of eyeRefinedIrisCenter
	if((calibratedChessBoardPtsLast==calibratedChessBoardPts) && getCalibrationPoints && (eyeState == Eyeopen )
		&& setEyeCornerAndEyePosReady){		
			if(countEyePut>5){
				MedianFilter1DProcedure(posQueueX , median_filter1DEyeRefinedCenter_size 
														, eyeRefinedIrisCenter.x , filteredEyeRefinedCenter.x);	
				MedianFilter1DProcedure(posQueueY , median_filter1DEyeRefinedCenter_size 
														, eyeRefinedIrisCenter.y , filteredEyeRefinedCenter.y);			

			}
			++countEyePut;
	}



	++count_calProcedure;
	time_duringLast = time_during;
	calibratedChessBoardPtsLast = calibratedChessBoardPts;
	if(calibration_CollectDataDone){	
		if(calibrationPts_space==calibrationPattern::Step_space_two){
			AnchorPointsCollenction(numXSpace , numYSpace , calInterAnchorPoints);
		}

		////Draw Chessboard for report
		//Mat ChessBoardThesisPic = Scene_chessboard.clone();
		//char test_imageFille[MAX_WORD_LEN];
		//sprintf(test_imageFille , "%s\\ChessBoard_%d.jpg" , analysisGazeOutputDir , calibrationPts_space);
		//for(int i=0;i<calibratedCalPoints.size();++i){
		//	circle(ChessBoardThesisPic , calibratedCalPoints[i] , MIN(x_dist , y_dist)/2-5 , Scalar(150 , 200 , 50) , cv::FILLED);
		//	Draw_Cross(ChessBoardThesisPic , calibratedCalPoints[i].x , calibratedCalPoints[i].y 
		//						, MIN(x_dist , y_dist)/10 , MIN(x_dist , y_dist)/10 , Scalar(0 , 10 , 255) , 2);
		//}
		//imwrite(test_imageFille , ChessBoardThesisPic);

		calibrationProcedureBegin = false;				
	}
	return true;
}

inline void GazeEstimation_PolyNomial(Point &gazePoint , const Point &eyePosPoint 
														, const Mat &EyePtsTransformMat , const Mat &ScenePtstransformMat
														, const int &n_order 
														, const double *const mapping_paramX , const double *const mapping_paramY)
{
	double est_x = 0;
	double est_y = 0;

	Mat EyeNormalzed(3 , 1 , CV_64F);
	Mat EyeOriginal(3 , 1 , CV_64F);
	Mat SceneNormalized(3 , 1 , CV_64F);
	Mat SceneDeNormalized(3 , 1 , CV_64F);
	Mat A_CoeffMatrix(1 , numberOfVar , CV_64F);

	EyeOriginal.at<double>(0 , 0) = eyePosPoint.x;
	EyeOriginal.at<double>(1 , 0) = eyePosPoint.y;
	EyeOriginal.at<double>(2 , 0) = 1;
	EyeNormalzed = EyePtsTransformMat*EyeOriginal;
		
	int pos_horiz = 0;
	for(int p=1;p<n_order+1;++p){
		for(int k=0;k<p+1;++k){				
			A_CoeffMatrix.at<double>(0 , pos_horiz) = powf(EyeNormalzed.at<double>(0 , 0) , p-k)*powf(EyeNormalzed.at<double>(1 , 0) , k);				
			++pos_horiz;
		}
	}	
	A_CoeffMatrix.at<double>(0 , numberOfVar - 1) = 1;

	for(int i=0;i<numberOfVar;++i){
		est_x+= mapping_paramX[i]*A_CoeffMatrix.at<double>(0 , i);
		est_y+= mapping_paramY[i]*A_CoeffMatrix.at<double>(0 , i);
	}
	
	SceneNormalized.at<double>(0 , 0) = est_x;
	SceneNormalized.at<double>(1 , 0) = est_y;
	SceneNormalized.at<double>(2 , 0) = 1;
		
	SceneDeNormalized = ScenePtstransformMat.inv()*SceneNormalized;
	gazePoint.x = SceneDeNormalized.at<double>(0 , 0)/SceneDeNormalized.at<double>(2 , 0);
	gazePoint.y = SceneDeNormalized.at<double>(1 , 0)/SceneDeNormalized.at<double>(2 , 0);
}
//
//inline void GazeEstimation_SVR(Point &gazePoint , const Point &eyePosPoint 
//												,const dlib::decision_function<kernel_type> &svr_model_X
//												,const dlib::decision_function<kernel_type> &svr_model_Y)
//{
//	sample_type eyeInput;
//	eyeInput(0 , 0) = eyePosPoint.x;
//	eyeInput(1 , 0) = eyePosPoint.y;
//
//	gazePoint.x = svr_model_X(eyeInput);
//	gazePoint.y = svr_model_Y(eyeInput);
//}


inline void FindNearestFourPts(Point &left_InterCalPts_eyeRegion , Point &right_InterCalPts_eyeRegion
											, Point &up_InterCalPts_eyeRegion , Point &down_InterCalPts_eyeRegion	
											, Point &left_InterCalPts_gazeRegion , Point &right_InterCalPts_gazeRegion
											, Point &up_InterCalPts_gazeRegion , Point &down_InterCalPts_gazeRegion	
											, const std::vector<Point> &calibratedEyeRefinedCenter
											, const std::vector<Point> &calibratedCalPoints
											, const Point &eyePosPoint
											, const std::vector<sliceMapElement> &mappingSliceMap
											, const bool &isChessBoardSideOdd
											, bool &notInEyeRegion
											, bool &YInterpolation , bool &XInterpolation)
{	
	int mapIndex = -1;
	bool findIndex = false;	
	for(int i=0;i<mappingSliceMap.size();++i){
		if(mappingSliceMap[i].returnMap().at<uchar>(eyePosPoint.y , eyePosPoint.x)==255){			
			mapIndex = i;
			findIndex = true;
			break;
		}
	}
	
	if(findIndex){//Linear Interpolation
		notInEyeRegion = false;
		YInterpolation = true;
		XInterpolation = true;
	}else{//The input eyePosPoint is not in the calibration region
		//Linear Extrapolation
		int nearestMapIndex = -1;
		float minDist = FLT_MAX;
		for(int i=0;i<mappingSliceMap.size();++i){			
			float dist = DistanceCaculateEuclidean(mappingSliceMap[i].returnCenter() , eyePosPoint);			
			if(dist<minDist){
				minDist = dist;
				nearestMapIndex = i;
			}
		}
		mapIndex = nearestMapIndex;
		notInEyeRegion = true;		
	}

	
	if(isChessBoardSideOdd){//odd X odd		
		if(mappingSliceMap[mapIndex].returnDefect("up")==true){//left right down			
			pair<float , float> upLineEq = mappingSliceMap[mapIndex].returnLineEq();

			left_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("left").first;
			left_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("left").second];

			right_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("right").first;
			right_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("right").second];

			down_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("down").first;
			down_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("down").second];

			up_InterCalPts_eyeRegion.x = eyePosPoint.x;
			up_InterCalPts_eyeRegion.y = upLineEq.first*eyePosPoint.x + upLineEq.second;
			up_InterCalPts_gazeRegion.x = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("down").second].x;
			up_InterCalPts_gazeRegion.y = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("left").second].y;
		}else if(mappingSliceMap[mapIndex].returnDefect("right")==true){//up right down			
			pair<float , float> leftLineEq = mappingSliceMap[mapIndex].returnLineEq();

			up_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("up").first;
			up_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("up").second];

			right_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("right").first;
			right_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("right").second];

			down_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("down").first;
			down_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("down").second];

			left_InterCalPts_eyeRegion.x = (eyePosPoint.y - leftLineEq.second)/float(leftLineEq.first);
			left_InterCalPts_eyeRegion.y = eyePosPoint.y;
			left_InterCalPts_gazeRegion.x = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("down").second].x;
			left_InterCalPts_gazeRegion.y = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("right").second].y;
		}else if(mappingSliceMap[mapIndex].returnDefect("down")==true){			
			pair<float , float> downLineEq = mappingSliceMap[mapIndex].returnLineEq();

			up_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("up").first;
			up_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("up").second];

			right_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("right").first;
			right_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("right").second];

			left_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("left").first;
			left_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("left").second];

			down_InterCalPts_eyeRegion.x = eyePosPoint.x;
			down_InterCalPts_eyeRegion.y = downLineEq.first*eyePosPoint.x + downLineEq.second;
			down_InterCalPts_gazeRegion.x = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("up").second].x;
			down_InterCalPts_gazeRegion.y = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("left").second].y;			
		}else if(mappingSliceMap[mapIndex].returnDefect("left")==true){			
			pair<float , float> rightLineEq = mappingSliceMap[mapIndex].returnLineEq();

			up_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("up").first;
			up_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("up").second];

			left_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("left").first;
			left_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("left").second];

			down_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("down").first;
			down_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("down").second];

			right_InterCalPts_eyeRegion.x = (eyePosPoint.y - rightLineEq.second)/float(rightLineEq.first);
			right_InterCalPts_eyeRegion.y = eyePosPoint.y;
			right_InterCalPts_gazeRegion.x = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("up").second].x;
			right_InterCalPts_gazeRegion.y = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("left").second].y;			
		}else{//no defects			
			up_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("up").first;
			up_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("up").second];

			left_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("left").first;
			left_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("left").second];

			down_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("down").first;
			down_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("down").second];

			right_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("right").first;
			right_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("right").second];			
		}
	}else{//even X even
		if(mappingSliceMap[mapIndex].returnDefect("up")==true && mappingSliceMap[mapIndex].returnDefect("right")==true){
			up_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("right").first;
			up_InterCalPts_gazeRegion.x = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("down").second].x;
			up_InterCalPts_gazeRegion.y = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("right").second].y;				

			left_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("down").first;
			left_InterCalPts_gazeRegion = up_InterCalPts_gazeRegion;

			down_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("down").first;
			down_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("down").second];

			right_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("right").first;
			right_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("right").second];	
		}else if(mappingSliceMap[mapIndex].returnDefect("left")==true && mappingSliceMap[mapIndex].returnDefect("down")==true){
			up_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("up").first;
			up_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("up").second];

			left_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("left").first;
			left_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("left").second];

			down_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("left").first;
			down_InterCalPts_gazeRegion.x = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("up").second].x;
			down_InterCalPts_gazeRegion.y = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("left").second].y;

			right_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("up").first;
			right_InterCalPts_gazeRegion = down_InterCalPts_gazeRegion;
		}else if(mappingSliceMap[mapIndex].returnDefect("up")==true){//left right down
			pair<float , float> upLineEq = mappingSliceMap[mapIndex].returnLineEq();

			left_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("left").first;
			left_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("left").second];

			right_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("right").first;
			right_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("right").second];

			down_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("down").first;
			down_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("down").second];

			up_InterCalPts_eyeRegion.x = eyePosPoint.x;
			up_InterCalPts_eyeRegion.y = upLineEq.first*eyePosPoint.x + upLineEq.second;
			up_InterCalPts_gazeRegion.x = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("down").second].x;
			up_InterCalPts_gazeRegion.y = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("left").second].y;
		}else if(mappingSliceMap[mapIndex].returnDefect("right")==true){//up right down
			pair<float , float> leftLineEq = mappingSliceMap[mapIndex].returnLineEq();

			up_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("up").first;
			up_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("up").second];

			right_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("right").first;
			right_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("right").second];

			down_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("down").first;
			down_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("down").second];

			left_InterCalPts_eyeRegion.x = (eyePosPoint.y - leftLineEq.second)/float(leftLineEq.first);
			left_InterCalPts_eyeRegion.y = eyePosPoint.y;
			left_InterCalPts_gazeRegion.x = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("down").second].x;
			left_InterCalPts_gazeRegion.y = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("right").second].y;
		}else if(mappingSliceMap[mapIndex].returnDefect("down")==true){
			pair<float , float> downLineEq = mappingSliceMap[mapIndex].returnLineEq();

			up_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("up").first;
			up_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("up").second];

			right_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("right").first;
			right_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("right").second];

			left_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("left").first;
			left_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("left").second];

			down_InterCalPts_eyeRegion.x = eyePosPoint.x;
			down_InterCalPts_eyeRegion.y = downLineEq.first*eyePosPoint.x + downLineEq.second;
			down_InterCalPts_gazeRegion.x = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("left").second].x;
			down_InterCalPts_gazeRegion.y = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("up").second].y;			
		}else if(mappingSliceMap[mapIndex].returnDefect("left")==true){
			pair<float , float> rightLineEq = mappingSliceMap[mapIndex].returnLineEq();

			up_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("up").first;
			up_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("up").second];

			left_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("left").first;
			left_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("left").second];

			down_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("down").first;
			down_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("down").second];

			right_InterCalPts_eyeRegion.x = (eyePosPoint.y - rightLineEq.second)/float(rightLineEq.first);
			right_InterCalPts_eyeRegion.y = eyePosPoint.y;
			right_InterCalPts_gazeRegion.x = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("up").second].x;
			right_InterCalPts_gazeRegion.y = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("left").second].y;			
		}else{//no defects
			up_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("up").first;
			up_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("up").second];

			left_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("left").first;
			left_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("left").second];

			down_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("down").first;
			down_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("down").second];

			right_InterCalPts_eyeRegion = mappingSliceMap[mapIndex].returnInterpolationPairs("right").first;
			right_InterCalPts_gazeRegion = calibratedCalPoints[mappingSliceMap[mapIndex].returnInterpolationPairs("right").second];			
		}		
	}//end else even X even	

	
	if(notInEyeRegion){
		if(eyePosPoint.x<right_InterCalPts_eyeRegion.x && eyePosPoint.x>left_InterCalPts_eyeRegion.x){
			XInterpolation = true;
		}else{
			XInterpolation = false;
		}
		
		if(eyePosPoint.y>up_InterCalPts_eyeRegion.y && eyePosPoint.y<down_InterCalPts_eyeRegion.y){
			YInterpolation = true;
		}else{
			YInterpolation = false;
		}	
	}
}

inline void GazeEstimation_BilinearInterpolation(Point &gazePoint , const Point &eyePosPoint 
																		, const std::vector<Point> &calibratedEyeRefinedCenter
																		, const std::vector<Point> &calibratedCalPoints
																		, const std::vector<sliceMapElement> &mappingSliceMap
																		, const bool &isChessBoardSideOdd)
{
	Point left_InterCalPts_eyeRegion , left_InterCalPts_gazeRegion;
	Point right_InterCalPts_eyeRegion , right_InterCalPts_gazeRegion;
	Point up_InterCalPts_eyeRegion , up_InterCalPts_gazeRegion;
	Point down_InterCalPts_eyeRegion , down_InterCalPts_gazeRegion;
	bool notInEyeRegion = false;
	bool YInterpolation = true;
	bool XInterpolation = true;

	
	FindNearestFourPts(left_InterCalPts_eyeRegion , right_InterCalPts_eyeRegion
									, up_InterCalPts_eyeRegion , down_InterCalPts_eyeRegion	
									, left_InterCalPts_gazeRegion , right_InterCalPts_gazeRegion
									, up_InterCalPts_gazeRegion , down_InterCalPts_gazeRegion	
									, calibratedEyeRefinedCenter
									, calibratedCalPoints
									, eyePosPoint
									, mappingSliceMap
									, isChessBoardSideOdd
									, notInEyeRegion
									, YInterpolation , XInterpolation);
	
	float denominatorX = float(right_InterCalPts_eyeRegion.x - left_InterCalPts_eyeRegion.x);
	float coeffLeft = (right_InterCalPts_eyeRegion.x - eyePosPoint.x)/denominatorX;
	float coeffRight = (eyePosPoint.x - left_InterCalPts_eyeRegion.x)/denominatorX;

	float denominatorY = float(down_InterCalPts_eyeRegion.y - up_InterCalPts_eyeRegion.y);
	float coeffUp = (down_InterCalPts_eyeRegion.y - eyePosPoint.y)/denominatorY;
	float coeffDown = (eyePosPoint.y - up_InterCalPts_eyeRegion.y)/denominatorY;

	gazePoint.x = coeffLeft*left_InterCalPts_gazeRegion.x + coeffRight*right_InterCalPts_gazeRegion.x;
	gazePoint.y = coeffUp*up_InterCalPts_gazeRegion.y + coeffDown*down_InterCalPts_gazeRegion.y;	
}


inline void GazeEstimation_HomographySliceMapping(Point &gazePoint , const Point &eyePosPoint 																				
																				, const std::vector<sliceMapElement> &mappingSliceMap
																				, const std::vector<Point> &calibratedCalPoints/*Debug*/)
{
	Mat HompgraphyMatrix;
	Mat SceneMat(3 , 1 , CV_64F);
	Mat EyeOriginal(3 , 1 , CV_64F);
	int mapIndex = -1;	
	bool findIndex = false;	

	EyeOriginal.at<double>(0 , 0) = eyePosPoint.x;
	EyeOriginal.at<double>(1 , 0) = eyePosPoint.y;
	EyeOriginal.at<double>(2 , 0) = 1;

	for(int i=0;i<mappingSliceMap.size();++i){
		if(mappingSliceMap[i].returnMap().at<uchar>(eyePosPoint.y , eyePosPoint.x)==255){			
			mapIndex = i;
			findIndex = true;
			break;
		}
	}
		
	if(!findIndex){//The input eyePosPoint is not in the calibration region
		//Find the nearest
		int nearestMapIndex = -1;
		float minDist = FLT_MAX;
		for(int i=0;i<mappingSliceMap.size();++i){			
			float dist = DistanceCaculateEuclidean(mappingSliceMap[i].returnCenter() , eyePosPoint);			
			if(dist<minDist){
				minDist = dist;
				nearestMapIndex = i;
			}
		}
		mapIndex = nearestMapIndex;	
	}

	HompgraphyMatrix = mappingSliceMap[mapIndex].returnHompgraphyMatrix();
	SceneMat = HompgraphyMatrix*EyeOriginal;
	gazePoint.x = SceneMat.at<double>(0 , 0)/SceneMat.at<double>(2 , 0);
	gazePoint.y = SceneMat.at<double>(1 , 0)/SceneMat.at<double>(2 , 0);
}

inline void SearchEyeIndex(const Point &searchPoint , const vector<Point> &calibratedCalPoints , int &index_return){
	for(int i=0;i<calibratedCalPoints.size();++i){
		if(searchPoint==calibratedCalPoints[i]){
			index_return = i;
			break;
		}
	}	
}

inline bool BuildSlicingMap(const std::vector<Point> &calInterAnchorPoints 
										, const std::vector<Point> &calibratedEyeRefinedCenter , const std::vector<Point> &calibratedCalPoints
										, const int &count_line_cols , const int &count_line_rows
										, const int &x_dist , const int &y_dist
										, std::vector<sliceMapElement> &mappingSliceMap
										, const int &calBrationMethod
										, const bool &isChessBoardSideOdd)
{	
	if(calibratedEyeRefinedCenter.size()<calibratedCalPoints.size() || calibratedCalPoints.size()==0 || calibratedEyeRefinedCenter.size()==0){
		printf("\n No Eye Pts in BuildSlicingMap !\n");
		return false;
	}
	
	int x_initial = x_dist/2;
	int y_initial = y_dist/2;
	for(int i=0;i<calInterAnchorPoints.size();++i){
		sliceMapElement sliceMap;
		if(isChessBoardSideOdd){//count_line_cols & count_line_rows are odd			
			if(calBrationMethod==calibrationMethod::HomographySliceMapping){//Homography Slicing Method	
				int leftIndex;
				int upIndex;
				int rightIndex;
				int downIndex;
				Point leftSearchPt;
				Point upSearchPt;
				Point rightSearchPt;
				Point downSearchPt;
				if(calInterAnchorPoints[i].y -1<0){//up defects								
					leftSearchPt.x = x_initial + (calInterAnchorPoints[i].x - 1)*x_dist;
					leftSearchPt.y = y_initial + calInterAnchorPoints[i].y*y_dist;

					upSearchPt.x = x_initial + calInterAnchorPoints[i].x*x_dist;
					upSearchPt.y = y_initial + calInterAnchorPoints[i].y*y_dist;

					rightSearchPt.x = x_initial + (calInterAnchorPoints[i].x + 1)*x_dist;
					rightSearchPt.y = y_initial + calInterAnchorPoints[i].y*y_dist;

					downSearchPt.x = x_initial + calInterAnchorPoints[i].x*x_dist;
					downSearchPt.y = y_initial + (calInterAnchorPoints[i].y+1)*y_dist;	
				}else if(calInterAnchorPoints[i].x +1>count_line_cols - 1){//right defects					
					leftSearchPt.x = x_initial + (calInterAnchorPoints[i].x - 1)*x_dist;
					leftSearchPt.y = y_initial + calInterAnchorPoints[i].y*y_dist;

					upSearchPt.x = x_initial + calInterAnchorPoints[i].x*x_dist;
					upSearchPt.y = y_initial + (calInterAnchorPoints[i].y - 1)*y_dist;
				
					rightSearchPt.x = x_initial + calInterAnchorPoints[i].x*x_dist;
					rightSearchPt.y = y_initial + (calInterAnchorPoints[i].y)*y_dist;

					downSearchPt.x = x_initial + calInterAnchorPoints[i].x*x_dist;
					downSearchPt.y = y_initial + (calInterAnchorPoints[i].y+1)*y_dist;			
				}else if(calInterAnchorPoints[i].y +1> count_line_rows - 1){//down defects				
					leftSearchPt.x = x_initial + (calInterAnchorPoints[i].x - 1)*x_dist;
					leftSearchPt.y = y_initial + calInterAnchorPoints[i].y*y_dist;

					upSearchPt.x = x_initial + calInterAnchorPoints[i].x*x_dist;
					upSearchPt.y = y_initial + (calInterAnchorPoints[i].y - 1)*y_dist;

					rightSearchPt.x = x_initial + (calInterAnchorPoints[i].x + 1)*x_dist;
					rightSearchPt.y = y_initial + calInterAnchorPoints[i].y*y_dist;

					downSearchPt.x = x_initial + calInterAnchorPoints[i].x*x_dist;
					downSearchPt.y = y_initial + calInterAnchorPoints[i].y*y_dist;		
				}else if(calInterAnchorPoints[i].x - 1 <0){//left defects					
					leftSearchPt.x = x_initial + calInterAnchorPoints[i].x*x_dist;
					leftSearchPt.y = y_initial + calInterAnchorPoints[i].y*y_dist;

					upSearchPt.x = x_initial + calInterAnchorPoints[i].x*x_dist;
					upSearchPt.y = y_initial + (calInterAnchorPoints[i].y - 1)*y_dist;

					rightSearchPt.x = x_initial + (calInterAnchorPoints[i].x + 1)*x_dist;
					rightSearchPt.y = y_initial + calInterAnchorPoints[i].y*y_dist;

					downSearchPt.x = x_initial + calInterAnchorPoints[i].x*x_dist ;
					downSearchPt.y = y_initial + (calInterAnchorPoints[i].y + 1)*y_dist;
				}else{//No defects
					leftSearchPt.x = x_initial + (calInterAnchorPoints[i].x - 1)*x_dist;
					leftSearchPt.y = y_initial + calInterAnchorPoints[i].y*y_dist;

					upSearchPt.x = x_initial + calInterAnchorPoints[i].x*x_dist;
					upSearchPt.y = y_initial + (calInterAnchorPoints[i].y - 1)*y_dist;

					rightSearchPt.x = x_initial + (calInterAnchorPoints[i].x + 1)*x_dist;
					rightSearchPt.y = y_initial + calInterAnchorPoints[i].y*y_dist;

					downSearchPt.x = x_initial + calInterAnchorPoints[i].x*x_dist;
					downSearchPt.y = y_initial + (calInterAnchorPoints[i].y + 1)*y_dist;	
				}	

				SearchEyeIndex(leftSearchPt , calibratedCalPoints , leftIndex);
				SearchEyeIndex(upSearchPt , calibratedCalPoints , upIndex);
				SearchEyeIndex(rightSearchPt , calibratedCalPoints , rightIndex);
				SearchEyeIndex(downSearchPt , calibratedCalPoints , downIndex);		
				sliceMap.setAnchorPtsALL(pair<Point , int>(calibratedEyeRefinedCenter[upIndex] , upIndex)
														, pair<Point , int>(calibratedEyeRefinedCenter[downIndex] , downIndex) 
														, pair<Point , int>(calibratedEyeRefinedCenter[leftIndex] , leftIndex)
														, pair<Point , int>(calibratedEyeRefinedCenter[rightIndex] , rightIndex));		
			}else{//Other Method				
				if(calInterAnchorPoints[i].y -1<0){//up defects								
					Point leftSearchPt(x_initial + (calInterAnchorPoints[i].x - 1)*x_dist , y_initial + calInterAnchorPoints[i].y*y_dist);
					Point rightSearchPt(x_initial + (calInterAnchorPoints[i].x + 1)*x_dist , y_initial + calInterAnchorPoints[i].y*y_dist);
					Point downSearchPt(x_initial + (calInterAnchorPoints[i].x)*x_dist , y_initial + (calInterAnchorPoints[i].y+1)*y_dist);
					int leftIndex;
					int rightIndex;
					int downIndex;

					SearchEyeIndex(leftSearchPt , calibratedCalPoints , leftIndex);
					SearchEyeIndex(rightSearchPt , calibratedCalPoints , rightIndex);
					SearchEyeIndex(downSearchPt , calibratedCalPoints , downIndex);
					sliceMap.setAnchorPtsUpDelete(pair<Point , int>(calibratedEyeRefinedCenter[downIndex] , downIndex)
																	, pair<Point , int>(calibratedEyeRefinedCenter[leftIndex] , leftIndex)
																	, pair<Point , int>(calibratedEyeRefinedCenter[rightIndex] , rightIndex));					
				}else if(calInterAnchorPoints[i].x +1>count_line_cols - 1){//right defects					
					Point leftSearchPt(x_initial + (calInterAnchorPoints[i].x - 1)*x_dist , y_initial + calInterAnchorPoints[i].y*y_dist);
					Point upSearchPt(x_initial + (calInterAnchorPoints[i].x)*x_dist , y_initial + (calInterAnchorPoints[i].y - 1)*y_dist);
					Point downSearchPt(x_initial + (calInterAnchorPoints[i].x)*x_dist , y_initial + (calInterAnchorPoints[i].y+1)*y_dist);
					int leftIndex;
					int upIndex;
					int downIndex;

					SearchEyeIndex(leftSearchPt , calibratedCalPoints , leftIndex);
					SearchEyeIndex(upSearchPt , calibratedCalPoints , upIndex);
					SearchEyeIndex(downSearchPt , calibratedCalPoints , downIndex);
					sliceMap.setAnchorPtsRightDelete(pair<Point , int>(calibratedEyeRefinedCenter[upIndex] , upIndex)
																	, pair<Point , int>(calibratedEyeRefinedCenter[downIndex] , downIndex)
																	, pair<Point , int>(calibratedEyeRefinedCenter[leftIndex] , leftIndex));						
				}else if(calInterAnchorPoints[i].y +1> count_line_rows - 1){//down defects				
					Point leftSearchPt(x_initial + (calInterAnchorPoints[i].x - 1)*x_dist , y_initial + calInterAnchorPoints[i].y*y_dist);
					Point upSearchPt(x_initial + (calInterAnchorPoints[i].x)*x_dist , y_initial + (calInterAnchorPoints[i].y - 1)*y_dist);
					Point rightSearchPt(x_initial + (calInterAnchorPoints[i].x + 1)*x_dist , y_initial + (calInterAnchorPoints[i].y)*y_dist);
					int leftIndex;
					int upIndex;
					int rightIndex;

					SearchEyeIndex(leftSearchPt , calibratedCalPoints , leftIndex);
					SearchEyeIndex(upSearchPt , calibratedCalPoints , upIndex);
					SearchEyeIndex(rightSearchPt , calibratedCalPoints , rightIndex);
					sliceMap.setAnchorPtsDownDelete(pair<Point , int>(calibratedEyeRefinedCenter[upIndex] , upIndex)
																	, pair<Point , int>(calibratedEyeRefinedCenter[leftIndex] , leftIndex)
																	, pair<Point , int>(calibratedEyeRefinedCenter[rightIndex] , rightIndex));					
				}else if(calInterAnchorPoints[i].x - 1 <0){//left defects					
					Point upSearchPt(x_initial + (calInterAnchorPoints[i].x)*x_dist , y_initial + (calInterAnchorPoints[i].y - 1)*y_dist);
					Point rightSearchPt(x_initial + (calInterAnchorPoints[i].x + 1)*x_dist , y_initial + (calInterAnchorPoints[i].y)*y_dist);
					Point downSearchPt(x_initial + (calInterAnchorPoints[i].x)*x_dist , y_initial + (calInterAnchorPoints[i].y + 1)*y_dist);
					int upIndex;
					int rightIndex;
					int downIndex;

					SearchEyeIndex(upSearchPt , calibratedCalPoints , upIndex);
					SearchEyeIndex(rightSearchPt , calibratedCalPoints , rightIndex);
					SearchEyeIndex(downSearchPt , calibratedCalPoints , downIndex);
					sliceMap.setAnchorPtsLeftDelete(pair<Point , int>(calibratedEyeRefinedCenter[upIndex] , upIndex)
																	, pair<Point , int>(calibratedEyeRefinedCenter[downIndex] , downIndex)
																	, pair<Point , int>(calibratedEyeRefinedCenter[rightIndex] , rightIndex));		
				}else{//No defects
					Point leftSearchPt(x_initial + (calInterAnchorPoints[i].x - 1)*x_dist , y_initial + (calInterAnchorPoints[i].y)*y_dist);
					Point upSearchPt(x_initial + (calInterAnchorPoints[i].x)*x_dist , y_initial + (calInterAnchorPoints[i].y - 1)*y_dist);
					Point rightSearchPt(x_initial + (calInterAnchorPoints[i].x + 1)*x_dist , y_initial + (calInterAnchorPoints[i].y)*y_dist);
					Point downSearchPt(x_initial + (calInterAnchorPoints[i].x)*x_dist , y_initial + (calInterAnchorPoints[i].y + 1)*y_dist);
					int leftIndex;
					int upIndex;
					int rightIndex;
					int downIndex;
					SearchEyeIndex(leftSearchPt , calibratedCalPoints , leftIndex);
					SearchEyeIndex(upSearchPt , calibratedCalPoints , upIndex);
					SearchEyeIndex(rightSearchPt , calibratedCalPoints , rightIndex);
					SearchEyeIndex(downSearchPt , calibratedCalPoints , downIndex);		
					sliceMap.setAnchorPtsALL(pair<Point , int>(calibratedEyeRefinedCenter[upIndex] , upIndex)
															, pair<Point , int>(calibratedEyeRefinedCenter[downIndex] , downIndex) 
															, pair<Point , int>(calibratedEyeRefinedCenter[leftIndex] , leftIndex)
															, pair<Point , int>(calibratedEyeRefinedCenter[rightIndex] , rightIndex));	
					
				}
			}//End Other Method			
		}else{//count_line_cols & count_line_rows are even
			if(calInterAnchorPoints[i].y -1< 0 && calInterAnchorPoints[i].x +1> count_line_cols - 1){//up & right defects	
				Point leftSearchPt(x_initial + (calInterAnchorPoints[i].x - 1)*x_dist , y_initial + (calInterAnchorPoints[i].y)*y_dist);					
				Point downSearchPt(x_initial + (calInterAnchorPoints[i].x)*x_dist , y_initial + (calInterAnchorPoints[i].y + 1)*y_dist);
				int leftIndex;					
				int downIndex;
				SearchEyeIndex(leftSearchPt , calibratedCalPoints , leftIndex);				
				SearchEyeIndex(downSearchPt , calibratedCalPoints , downIndex);		

				sliceMap.setAnchorPtsUpRightDelete(pair<Point , int>(calibratedEyeRefinedCenter[downIndex] , downIndex) 
																		, pair<Point , int>(calibratedEyeRefinedCenter[leftIndex] , leftIndex));
			}else if(calInterAnchorPoints[i].x -1< 0 && calInterAnchorPoints[i].y +1> count_line_rows - 1){//left & down defects				
				Point upSearchPt(x_initial + (calInterAnchorPoints[i].x)*x_dist , y_initial + (calInterAnchorPoints[i].y - 1)*y_dist);
				Point rightSearchPt(x_initial + (calInterAnchorPoints[i].x + 1)*x_dist , y_initial + (calInterAnchorPoints[i].y)*y_dist);								
				int upIndex;
				int rightIndex;
				
				SearchEyeIndex(upSearchPt , calibratedCalPoints , upIndex);
				SearchEyeIndex(rightSearchPt , calibratedCalPoints , rightIndex);
				sliceMap.setAnchorPtsDownLeftDelete(pair<Point , int>(calibratedEyeRefinedCenter[upIndex] , upIndex)																		
																			, pair<Point , int>(calibratedEyeRefinedCenter[rightIndex] , rightIndex));	

			}else if(calInterAnchorPoints[i].y - 1< 0){//up defects
				Point leftSearchPt(x_initial + (calInterAnchorPoints[i].x - 1)*x_dist , y_initial + calInterAnchorPoints[i].y*y_dist);
				Point rightSearchPt(x_initial + (calInterAnchorPoints[i].x + 1)*x_dist , y_initial + calInterAnchorPoints[i].y*y_dist);
				Point downSearchPt(x_initial + (calInterAnchorPoints[i].x)*x_dist , y_initial + (calInterAnchorPoints[i].y+1)*y_dist);
				int leftIndex;
				int rightIndex;
				int downIndex;

				SearchEyeIndex(leftSearchPt , calibratedCalPoints , leftIndex);
				SearchEyeIndex(rightSearchPt , calibratedCalPoints , rightIndex);
				SearchEyeIndex(downSearchPt , calibratedCalPoints , downIndex);
				sliceMap.setAnchorPtsUpDelete(pair<Point , int>(calibratedEyeRefinedCenter[downIndex] , downIndex)
																, pair<Point , int>(calibratedEyeRefinedCenter[leftIndex] , leftIndex)
																, pair<Point , int>(calibratedEyeRefinedCenter[rightIndex] , rightIndex));				
			}else if(calInterAnchorPoints[i].x + 1>count_line_cols - 1){//right defects
				Point leftSearchPt(x_initial + (calInterAnchorPoints[i].x - 1)*x_dist , y_initial + calInterAnchorPoints[i].y*y_dist);
				Point upSearchPt(x_initial + (calInterAnchorPoints[i].x)*x_dist , y_initial + (calInterAnchorPoints[i].y - 1)*y_dist);
				Point downSearchPt(x_initial + (calInterAnchorPoints[i].x)*x_dist , y_initial + (calInterAnchorPoints[i].y+1)*y_dist);
				int leftIndex;
				int upIndex;
				int downIndex;

				SearchEyeIndex(leftSearchPt , calibratedCalPoints , leftIndex);
				SearchEyeIndex(upSearchPt , calibratedCalPoints , upIndex);
				SearchEyeIndex(downSearchPt , calibratedCalPoints , downIndex);
				sliceMap.setAnchorPtsRightDelete(pair<Point , int>(calibratedEyeRefinedCenter[upIndex] , upIndex)
																, pair<Point , int>(calibratedEyeRefinedCenter[downIndex] , downIndex)
																, pair<Point , int>(calibratedEyeRefinedCenter[leftIndex] , leftIndex));				
			}else if(calInterAnchorPoints[i].y + 1>count_line_rows - 1){//down defects
				Point leftSearchPt(x_initial + (calInterAnchorPoints[i].x - 1)*x_dist , y_initial + calInterAnchorPoints[i].y*y_dist);
				Point upSearchPt(x_initial + (calInterAnchorPoints[i].x)*x_dist , y_initial + (calInterAnchorPoints[i].y - 1)*y_dist);
				Point rightSearchPt(x_initial + (calInterAnchorPoints[i].x + 1)*x_dist , y_initial + (calInterAnchorPoints[i].y)*y_dist);
				int leftIndex;
				int upIndex;
				int rightIndex;

				SearchEyeIndex(leftSearchPt , calibratedCalPoints , leftIndex);
				SearchEyeIndex(upSearchPt , calibratedCalPoints , upIndex);
				SearchEyeIndex(rightSearchPt , calibratedCalPoints , rightIndex);
				sliceMap.setAnchorPtsDownDelete(pair<Point , int>(calibratedEyeRefinedCenter[upIndex] , upIndex)
																, pair<Point , int>(calibratedEyeRefinedCenter[leftIndex] , leftIndex)
																, pair<Point , int>(calibratedEyeRefinedCenter[rightIndex] , rightIndex));					
			}else if(calInterAnchorPoints[i].x - 1< 0){//left defects
				Point upSearchPt(x_initial + (calInterAnchorPoints[i].x)*x_dist , y_initial + (calInterAnchorPoints[i].y - 1)*y_dist);
				Point rightSearchPt(x_initial + (calInterAnchorPoints[i].x + 1)*x_dist , y_initial + (calInterAnchorPoints[i].y)*y_dist);
				Point downSearchPt(x_initial + (calInterAnchorPoints[i].x)*x_dist , y_initial + (calInterAnchorPoints[i].y + 1)*y_dist);
				int upIndex;
				int rightIndex;
				int downIndex;

				SearchEyeIndex(upSearchPt , calibratedCalPoints , upIndex);
				SearchEyeIndex(rightSearchPt , calibratedCalPoints , rightIndex);
				SearchEyeIndex(downSearchPt , calibratedCalPoints , downIndex);
				sliceMap.setAnchorPtsLeftDelete(pair<Point , int>(calibratedEyeRefinedCenter[upIndex] , upIndex)
																, pair<Point , int>(calibratedEyeRefinedCenter[downIndex] , downIndex)
																, pair<Point , int>(calibratedEyeRefinedCenter[rightIndex] , rightIndex));				
			}else{//No defects
				Point leftSearchPt(x_initial + (calInterAnchorPoints[i].x - 1)*x_dist , y_initial + (calInterAnchorPoints[i].y)*y_dist);
				Point upSearchPt(x_initial + (calInterAnchorPoints[i].x)*x_dist , y_initial + (calInterAnchorPoints[i].y - 1)*y_dist);
				Point rightSearchPt(x_initial + (calInterAnchorPoints[i].x + 1)*x_dist , y_initial + (calInterAnchorPoints[i].y)*y_dist);
				Point downSearchPt(x_initial + (calInterAnchorPoints[i].x)*x_dist , y_initial + (calInterAnchorPoints[i].y + 1)*y_dist);
				int leftIndex;
				int upIndex;
				int rightIndex;
				int downIndex;
				SearchEyeIndex(leftSearchPt , calibratedCalPoints , leftIndex);
				SearchEyeIndex(upSearchPt , calibratedCalPoints , upIndex);
				SearchEyeIndex(rightSearchPt , calibratedCalPoints , rightIndex);
				SearchEyeIndex(downSearchPt , calibratedCalPoints , downIndex);		
				sliceMap.setAnchorPtsALL(pair<Point , int>(calibratedEyeRefinedCenter[upIndex] , upIndex)
														, pair<Point , int>(calibratedEyeRefinedCenter[downIndex] , downIndex) 
														, pair<Point , int>(calibratedEyeRefinedCenter[leftIndex] , leftIndex)
														, pair<Point , int>(calibratedEyeRefinedCenter[rightIndex] , rightIndex));				
			}		
		}

		if(calBrationMethod==calibrationMethod::BilinearInterpolation){
			sliceMap.reorganize();
			sliceMap.lineEqConstruction();			
		}
		
		sliceMap.mapConstruction();
		sliceMap.centerConstruction();			

		mappingSliceMap.push_back(sliceMap);	
	}// end for every anchor point

	return true;
}


inline bool get_randomPts_Num(const int &max_num , const int &numofPts , int* &rand_num){
	int rand_index = 0;
	int r;	
	bool is_new = 1;
	rand_num = new int [max_num+1]();

	srand(time(NULL));
	if(max_num<numofPts-1){
		printf("Doesn't get enough eye-gaze points pairs for calibration precedure.\n");
		return false;
	}

	while (rand_index <= max_num){
		is_new = true;
		r = (int)((rand()*1.0/RAND_MAX) * max_num);
		for (int i = 0; i < rand_index; ++i){
			if (r == rand_num[i]) {
				is_new = false;
				break;
			}
		}
		if (is_new) {
			rand_num[rand_index] = r;
			++rand_index;
		}
	}
	return true;
}

inline void GazePtsMeanCaculate(const vector<Point> &in_gaze , Point2d& meanPts){
	meanPts.x = 0;
	meanPts.y = 0;
	for(int i=0;i<in_gaze.size();++i){
		meanPts.x+=in_gaze[i].x;
		meanPts.y+=in_gaze[i].y;
	}
	meanPts.x/=in_gaze.size();
	meanPts.y/=in_gaze.size();	
}


inline void GazePtsStdDevCaculate(const vector<Point> &in_gaze , const Point& meanPts , double &stdDev){
	double stdDevSum = 0;
	for(int i=0;i<in_gaze.size();++i){
		stdDevSum+=powf(in_gaze[i].x - meanPts.x , 2.0)+powf(in_gaze[i].y - meanPts.y , 2.0);
	}
	stdDevSum/=in_gaze.size();
	stdDev = sqrtf(stdDevSum);
}

inline void CaculateGazeError(const vector<GazeTestingElement> &gazeTesting_vec){
	char outPutFileDir[MAX_WORD_LEN];	


	switch(testNumOfPts){
		case 9:{
			sprintf(outPutFileDir , "%s\\test_9TestPts" , analysisGazeOutputDir);
		}break;
		case 16:{
			sprintf(outPutFileDir , "%s\\test_16TestPts" , analysisGazeOutputDir);
		}break;
		case 25:{
			sprintf(outPutFileDir , "%s\\test_25TestPts" , analysisGazeOutputDir);
		}break;
	}


	char outFileName[MAX_WORD_LEN];
	sprintf(outFileName , "%s\\GazeTesting_error.txt" , outPutFileDir);
	ofstream fileErrOut(outFileName);		
	double sumErr = 0;
	Mat TestAll = Mat::zeros(testUsedH , testUsedW , CV_8UC3);	

	for(int i=0;i<gazeTesting_vec.size();++i){		
		double sumErr_element = 0;
		Point2d meanPts(0 , 0);
		double stdDev = 0;
		Mat Test = Mat::zeros(testUsedH , testUsedW , CV_8UC3);		
		Point testPts = gazeTesting_vec[i].returnTestPts();		
		vector<Point> gazePts = gazeTesting_vec[i].returnGazeVec();
		
		Draw_Cross(Test , testPts.x , testPts.y , 30 , 30 , Scalar(0 , 200 , 255) , 1);	
		Draw_Cross(TestAll , testPts.x , testPts.y , 30 , 30 , Scalar(0 , 200 , 255) , 1);		
		if(gazePts.size()==0){
			continue;
		}

		for(int j=0;j<gazePts.size();++j){			
			sumErr_element+=DistanceCaculateEuclidean(gazePts[j] , testPts);		
			Draw_Cross(Test , gazePts[j].x , gazePts[j].y , 15 , 15 , Scalar(200 , 255 , 10) , 1);				
			Draw_Cross(TestAll , gazePts[j].x , gazePts[j].y , 15 , 15 , Scalar(200 , 255 , 10) , 1);	
		}
		sumErr_element/=gazePts.size();
		sumErr+=sumErr_element;
		
		GazePtsMeanCaculate(gazePts , meanPts);
		GazePtsStdDevCaculate(gazePts , meanPts , stdDev);

		fileErrOut<<"i = "<<i<<" , err = "<<sumErr_element<<endl;
		if(sumErr_element>powf(10 , 7)){
			cout<<"i = "<<i<<" , err = "<<sumErr_element<<endl;
			cout<<"testPts = "<<testPts<<endl;
			for(int j=0;j<gazePts.size();++j){
				cout<<"j = "<<j<<" , pts = "<<gazePts[j]<<endl;
			}
			getchar();
		}

		char s[MAX_WORD_LEN];
		sprintf(s , "mean = (%3.2lf , %3.2lf)" , meanPts.x , meanPts.y);
		putText(Test
						, s
						, Point(50 , 100)
						, FONT_HERSHEY_DUPLEX
						, 1.5/480.0*Frame.cols
						, Scalar(0, 255 , 255)
						, 3
						,8
						, false );
		sprintf(s , "ground truth = (%d , %d)" , testPts.x , testPts.y);
		putText(Test
						, s
						, Point(50 , 300)
						, FONT_HERSHEY_DUPLEX
						, 1.5/480.0*Frame.cols
						, Scalar(0, 255 , 0)
						, 3
						,8
						, false );

		sprintf(s , "stdDev = %.5f" , stdDev);
		putText(Test
						, s
						, Point(50 , 200)
						, FONT_HERSHEY_DUPLEX
						, 1.5/480.0*Frame.cols
						, Scalar(0, 255 , 255)
						, 3
						,8
						, false );


		sprintf(outFileName , "%s\\GazeTesting_%d.jpg" , outPutFileDir , i);
		imwrite(outFileName , Test);		
	}//end for
	sprintf(outFileName , "%s\\GazeTesting_All.jpg" , outPutFileDir);
	imwrite(outFileName , TestAll);		

	sumErr/=gazeTesting_vec.size();
	fileErrOut<<"final , err = "<<sumErr<<" ("<<atan2(sumErr*pixelToCentiMeter , distanceOfMonitor)*ang_mul<<" degree)"<<endl;	
	printf("\nerr = %.5f (%.5f degree) \n" , sumErr , atan2(sumErr*pixelToCentiMeter , distanceOfMonitor)*ang_mul);
}

inline void EyeGazeTesting(int &countInEyeGazeTestingFunction , vector<Point> &testPtsVec , Mat &Scene_gazetest
										 , const int &x_distTest , const int &y_distTest , int *&rand_numTestPts 
										 , double &time_duringTest , double &time_duringCalLast
										 , const int &numXTestPts , const int &numYTestPts
										 , vector<GazeTestingElement> &gazeTesting_vec
										 , const Point &gazePoint 
										 , Point &dispTestPts , Point &dispTestPtsLast
										 , const bool &setEyeCornerAndEyePosReady
										 , const bool &eyeState
										 , bool &eyeGazeTestProcedure
										 , Mat &Scene_image
										 , int &countGazePut
										 , bool &realGazeInput)
{
	if(countInEyeGazeTestingFunction==0){		
		Mat TestAllPtsImg = Mat::zeros(testUsedH , testUsedW , CV_8UC3);
		int x_initial = x_distTest/2;
		int y_initial = y_distTest/2;

		for(int i=0;i<numYTestPts;++i){
			for(int j=0;j<numXTestPts;++j){
				Point tmp(x_initial+j*x_distTest , y_initial+i*y_distTest);
				testPtsVec.push_back(tmp);
				circle(TestAllPtsImg , tmp , MIN(x_distTest , y_distTest)/2-5 , Scalar(150 , 200 , 50) , cv::FILLED);
				Draw_Cross(TestAllPtsImg , tmp.x , tmp.y , MIN(x_distTest , y_distTest)/10 , MIN(x_distTest , y_distTest)/10 , Scalar(0 , 200 , 255) , 2);	
			}
		}			
		
		if(!get_randomPts_Num(testPtsVec.size() - 1 , testNumOfPts , rand_numTestPts)){
			return;
		}		

		Scene_gazetest = TestAllPtsImg.clone();
		imshow("Scene" , TestAllPtsImg);
		waitKey(0);		
		time_duringTest =  getTickCount();
	}

	int time_duringCal = (getTickCount() - time_duringTest)/getTickFrequency();				
	if((time_duringCal==testInterTimePoints && time_duringCal != time_duringCalLast) || countInEyeGazeTestingFunction==0){
		if(countInEyeGazeTestingFunction>=testNumOfPts){					
			CaculateGazeError(gazeTesting_vec);

			gazeTesting_vec.clear();
			eyeGazeTestProcedure = false;
			testPtsVec.clear();
			time_duringCalLast = -1;
			countInEyeGazeTestingFunction = 0;	
			return;
		}

		Scene_gazetest = Mat::zeros(testUsedH , testUsedW , CV_8UC3);	
		GazeTestingElement gazeTesting_element;

		dispTestPts = testPtsVec[rand_numTestPts[countInEyeGazeTestingFunction]];	

		gazeTesting_element.setTestPts(dispTestPts);
		gazeTesting_vec.push_back(gazeTesting_element);
		circle(Scene_gazetest , dispTestPts , MIN(x_distTest , y_distTest)/2-5 , Scalar(150 , 200 , 50) , cv::FILLED);
		Draw_Cross(Scene_gazetest , dispTestPts.x , dispTestPts.y , MIN(x_distTest , y_distTest)/10 , MIN(x_distTest , y_distTest)/10 , Scalar(0 , 200 , 255) , 2);		
				
		++countInEyeGazeTestingFunction;		
		countGazePut = 0;
		realGazeInput = false;
		imshow("Scene" , Scene_gazetest);	
		waitKey(0);	
		time_duringTest =  getTickCount();
	}


	if(dispTestPtsLast==dispTestPts && setEyeCornerAndEyePosReady==true && eyeState==Eyeopen){		
		if(countGazePut>2){
			if(checkpoint(gazePoint.x ,gazePoint.y , Scene_image)){
				gazeTesting_vec[countInEyeGazeTestingFunction-1].setCollectGazePts(gazePoint);					
			}
			realGazeInput = true;			
		}else{
			realGazeInput = false;
		}
		++countGazePut;		
	}

	dispTestPtsLast = dispTestPts;
	time_duringCalLast = time_duringCal;	
}

inline void ReadCameraCalibrationCoeff(Mat &CameraMatrix_Eye , Mat &DistCoeffs , const char *const fileName){
	ifstream in_file(fileName);	

	int count_x = 0;
	int count_y = 0;
	bool completeEye = false;
	string tmp;
	while(in_file>>tmp){
		if(tmp=="cameraMatrix" || tmp=="distCoeffs"){
			in_file>>tmp;
			if(tmp=="="){
				continue;
			}
		}else{
			if(!completeEye){
				istringstream iss(tmp);
				float v;
				iss>>dec>>v;			
				CameraMatrix_Eye.at<double>(count_y , count_x) = v;
				++count_x;
				if(count_x>CameraMatrix_Eye.cols-1){
					++count_y;
					count_x = 0;
				}	
				if(count_y>CameraMatrix_Eye.rows-1){
					count_y = 0;
					count_x = 0;
					completeEye = true;
				}	
			}else{
				istringstream iss(tmp);
				float v;
				iss>>dec>>v;			
				DistCoeffs.at<double>(count_y , count_x) = v;
				++count_x;
				if(count_x>DistCoeffs.cols-1){
					++count_y;
					count_x = 0;
				}	
				if(count_y>DistCoeffs.rows-1){
					count_y = 0;
					count_x = 0;
				}
			}
		
		}	
	}

	//cout<<"CameraMatrix_Eye = "<<endl;
	//for(int i=0;i<CameraMatrix_Eye.rows;++i){
	//	for(int j=0;j<CameraMatrix_Eye.cols;++j){
	//		cout<<CameraMatrix_Eye.at<double>(i , j)<<" ";
	//	}
	//	cout<<endl;
	//}
	//cout<<"DistCoeffs = "<<endl;
	//for(int i=0;i<DistCoeffs.rows;++i){
	//	for(int j=0;j<DistCoeffs.cols;++j){
	//		cout<<DistCoeffs.at<double>(i , j)<<" ";
	//	}
	//	cout<<endl;
	//}
	//getchar();
}

int main(int argc , char *argv[]){	
		//setNumThreads(1);
		//===========Parameter Setting==========/	
		/*Get Parameter List*/
		for(int i=1;i<argc;++i){
			if(!strcmp(argv[i],"-ct")){
				calibrationInterTimePoints = (int)atof(argv[i+1]);
			}else if(!strcmp(argv[i],"-mo")){
				calBrationMethod = calibrationMethod::Polynomial_Order;
				calibrationInterMappingFunctionOrder = (int)atof(argv[i+1]);
				orderInput = true;
			}else if(!strcmp(argv[i],"-cpl")){
				calPtsLength = (int)atof(argv[i+1]);
			}else if(!strcmp(argv[i],"-cps")){
				calibrationPts_space = (int)atof(argv[i+1]);
			}else if(!strcmp(argv[i],"-tt")){
				testInterTimePoints = (int)atof(argv[i+1]);
			}else if(!strcmp(argv[i],"-dom")){
				distanceOfMonitor = atof(argv[i+1]);
			}else if(!strcmp(argv[i],"-pcm")){
				pixelToCentiMeter = atof(argv[i+1]);
			}else if(!strcmp(argv[i],"-tn")){
				testNumOfPts = (int)atof(argv[i+1]);
			}else if(!strcmp(argv[i],"-imrf")){
				iris_maskModel_refreshFrame_Initial = (int)atof(argv[i+1]);
			}else if(!strcmp(argv[i],"-imaf")){
				iris_maskModel_refreshFrame_AfterGet = (int)atof(argv[i+1]);
			}else if(!strcmp(argv[i],"-imrt")){
				iris_colorModelValidTestingIrisRate_initial = atof(argv[i+1]);
			}else if(!strcmp(argv[i],"-imro")){
				iris_colorModelIrisRate_pixelInOthersOne = atof(argv[i+1]);
			}else if(!strcmp(argv[i] , "-tcase")){
				testSubject = true;
				sprintf(subjectFileName , "%s" , argv[i+1]);
			}else if(!strcmp(argv[i] , "-cfilename")){
				sprintf(calibrationNumTimesName, "%s", argv[i+1]);
			}else if (!strcmp(argv[i], "-testVarianceFolder")) {
				sprintf(testVarianceFolder, "%s", argv[i + 1]);
			}else if (!strcmp(argv[i], "-video_num")) {
				video_num = atof(argv[i + 1]);
			}else if (!strcmp(argv[i], "-thread_num")) {
				thread_num = (int)atof(argv[i + 1]);
			}
		}

		std::cout<<"calibrationInterTimePoints = "<<calibrationInterTimePoints<<endl;
		std::cout<<"calibrationInterMappingFunctionOrder = "<<calibrationInterMappingFunctionOrder<<endl;
		std::cout<<"calPtsLength = "<<calPtsLength<<endl;
		std::cout<<"calibrationPts_space = "<<calibrationPts_space<<endl;
		std::cout<<"testInterTimePoints = "<<testInterTimePoints<<endl;
		std::cout<<"distanceOfMonitor = "<<distanceOfMonitor<<endl;
		std::cout<<"pixelToCentiMeter = "<<pixelToCentiMeter<<endl;
		std::cout<<"testNumOfPts = "<<testNumOfPts<<endl;
		std::cout<<"iris_maskModel_refreshFrame_Initial = "<<iris_maskModel_refreshFrame_Initial<<endl;
		std::cout<<"iris_maskModel_refreshFrame_AfterGet = "<<iris_maskModel_refreshFrame_AfterGet<<endl;
		std::cout<<"iris_colorModelValidTestingIrisRate_initial = "<<iris_colorModelValidTestingIrisRate_initial<<endl;
		std::cout<<"iris_colorModelIrisRate_pixelInOthersOne = "<<iris_colorModelIrisRate_pixelInOthersOne <<endl;
		std::cout<<"calibrationNumTimesName = "<<calibrationNumTimesName <<endl;
		std::cout <<"testVarianceFolder = " << testVarianceFolder << endl;
		std::cout <<"video_num = " << video_num << endl;

		if(testSubject){
			std::cout<<"subjectFileName = "<<subjectFileName <<endl;			
		}
		//===========Read Camera Calibration Matrix==========/		
		Mat CameraMatrix_Eye = Mat::zeros(3, 3, CV_64F);
		Mat CameraMatrix_Scene = Mat::zeros(3, 3, CV_64F);
		Mat DistCoeffs_Eye = Mat::zeros(5, 1, CV_64F);
		Mat DistCoeffs_Scene = Mat::zeros(5, 1, CV_64F);
		
		ReadCameraCalibrationCoeff(CameraMatrix_Eye , DistCoeffs_Eye , ".\\txt\\outCameraCoeff_Eye2.txt");
		ReadCameraCalibrationCoeff(CameraMatrix_Scene , DistCoeffs_Scene , ".\\txt\\outCameraCoeff_Scene.txt");

		//===========Output File==========/					
		char tmpFileName[MAX_WORD_LEN];	
		char tmpCalName[MAX_WORD_LEN];	
		char tmpTestName[MAX_WORD_LEN];	
		char command[MAX_WORD_LEN];	

		if(testSubject){			
			sprintf(tmpFileName, "%s\\%s", analysisGazeOutputDir ,subjectFileName);
			struct stat buf;			

			//Create a new Subject file if not found
			if(stat(tmpCalName,&buf) != 0){//folder not found		
				sprintf(command , "md %s" , tmpFileName);
				system(command);
			}

			//Create a new CalibrationNum file
			sprintf(tmpCalName, "%s\\%s", tmpFileName ,calibrationNumTimesName);
			if(stat(tmpCalName,&buf) == 0){//folder found		
				sprintf(command , "rd /q /s %s" , tmpCalName);
				system(command);
				
				sprintf(command , "md %s" , tmpCalName);
				system(command);
			}else{//folder not found		
				sprintf(command , "md %s" , tmpCalName);
				system(command);
			}
			//Create a new test_16TestPts file
			sprintf(tmpTestName, "%s\\%s", tmpCalName ,"test_16TestPts");
			sprintf(command , "md %s" , tmpTestName);
			system(command);

			sprintf(analysisGazeOutputDir , "%s" , tmpCalName);
		}else{
			if(calibrationPts_space==calibrationPattern::Step_space_one){
				sprintf(tmpFileName , "%s\\Pattern_Step_One" , analysisGazeOutputDir);
				if(orderInput){
					sprintf(tmpFileName , "%s\\Order_%d_Cal_%d_Pts" , tmpFileName , calibrationInterMappingFunctionOrder , calPtsLength*calPtsLength);			
				}else{
					sprintf(tmpFileName , "%s\\Hybrid_Order_Cal_%d_Pts" , tmpFileName , calPtsLength*calPtsLength);
				}
				sprintf(analysisGazeOutputDir , "%s\\%s" , tmpFileName ,calibrationNumTimesName);
			}else{
				int fileNameNumber;
				switch(calPtsLength){
				case 3:{
					fileNameNumber = 5;
					}break;
				case 4:{
					fileNameNumber = 8;
					}break;
				case 5:{
					fileNameNumber = 13;
					}break;
				case 6:{
					fileNameNumber = 18;
					}break;
				case 7:{
					fileNameNumber = 25;
					}break;
				}
				sprintf(tmpFileName , "%s\\Pattern_Step_Two" , analysisGazeOutputDir);
				if(orderInput){
					sprintf(tmpFileName , "%s\\Order_%d_Cal_%d_Pts" , tmpFileName , calibrationInterMappingFunctionOrder , fileNameNumber);			
				}else{
					sprintf(tmpFileName , "%s\\Hybrid_Order_Cal_%d_Pts" , tmpFileName , fileNameNumber);
				}
				sprintf(analysisGazeOutputDir , "%s\\%s" , tmpFileName , calibrationNumTimesName);
			}
		}
		//===========Camera Choose==========/
		char video_input_file[MAX_WORD_LEN];
		sprintf(video_input_file, "C:\\Users\\Coslate\\Parallel_Programming\\Final_Project\\EyeGazeTrackingSystem_speedUp_vf\\EyeGazeTrackingSystem_speedUp_vf\\Test_Mean_Variance_Data\\med_down_%d\\med_down_%d.avi", video_num, video_num);
		std::cout << "video_input_file = " << video_input_file << std::endl;
		//VideoCapture cap(3);
		//VideoCapture cap(0);
		//VideoCapture cap_scene(1);
		//VideoCapture cap("C:\\Users\\Coslate\\Parallel_Programming\\Final_Project\\EyeGazeTrackingSystem_speedUp_vf\\EyeGazeTrackingSystem_speedUp_vf\\Test_Mean_Variance_Data\\med_down_15\\med_down_15.avi");
		VideoCapture cap(video_input_file);
		VideoCapture cap_scene(video_input_file);//This is temp setting for focusing only on eye image only. If one wants to do gaze estimation, this should set to scene video frame.


		//===========Input/Output File Set==========/
				
		namedWindow("EyePosition_CenterResult" , cv::WINDOW_AUTOSIZE);		
		//namedWindow("Scene" , cv::WINDOW_FULLSCREEN);
		//resizeWindow("Scene",MonitorW,MonitorH);//col , row
		//setWindowProperty("Scene", cv::WND_PROP_OPENGL, cv::WINDOW_FULLSCREEN);
		//moveWindow("Scene" , 0 , 0);
	
		//===========Write Video Out File Set==========/		
		char wirteOutCalVideoFileName[MAX_WORD_LEN];		
		char wirteOutTestVideoFileName[MAX_WORD_LEN];		
		char result_EyePosWriteOutFile[MAX_WORD_LEN];
		char result_BlinkWriteOutFile[MAX_WORD_LEN];
		char result_GazeWriteOutFile[MAX_WORD_LEN];
		char result_InputFrameOutFile[MAX_WORD_LEN];
		char result_WhiteBalanceWriteOutFile[MAX_WORD_LEN];
		char result_CoarseCenterWriteOutFile[MAX_WORD_LEN];
		char result_Fitting[MAX_WORD_LEN];
		char result_ROI[MAX_WORD_LEN];
		char result_Profiling [MAX_WORD_LEN];
		sprintf(wirteOutCalVideoFileName , "%s\\Calibration.avi" , analysisGazeOutputDir);
		sprintf(wirteOutTestVideoFileName , "%s\\GazeTest.avi" , analysisGazeOutputDir);
		sprintf(result_EyePosWriteOutFile , "%s\\%s\\EyePosResult.avi" , testVariancePtsDir, testVarianceFolder);
		sprintf(result_BlinkWriteOutFile , "%s\\BlinkResult.avi" , analysisGazeOutputDir);
		sprintf(result_GazeWriteOutFile , "%s\\GazeResult.avi" , analysisGazeOutputDir);
		sprintf(result_WhiteBalanceWriteOutFile , "%s\\%s\\WhiteBalance.avi" , testVariancePtsDir, testVarianceFolder);
		sprintf(result_InputFrameOutFile, "%s\\%s\\InputFrame.avi", testVariancePtsDir, testVarianceFolder);
		sprintf(result_Fitting, "%s\\%s\\Fitting.avi", testVariancePtsDir, testVarianceFolder);
		sprintf(result_ROI, "%s\\%s\\ROI.avi", testVariancePtsDir, testVarianceFolder);
		sprintf(result_CoarseCenterWriteOutFile , "%s\\CoarseCenter_FrontLight.avi" , analysisGazeOutputDir);
		
//#if writeResult
//		VideoWriter writer_Cal(
//			wirteOutCalVideoFileName
//			, CV_FOURCC('M', 'J', 'P', 'G'), 15.f, Size(calUsedW, calUsedH));// for saving frame
//
//		VideoWriter writer_GazeTesting(
//			wirteOutTestVideoFileName
//			, CV_FOURCC('M', 'J', 'P', 'G'), 15.f, Size(calUsedW, calUsedH));// for saving frame
//
//		VideoWriter writer_result_EyePos(
//			result_EyePosWriteOutFile
//			, CV_FOURCC('M', 'J', 'P', 'G'), 15.f, Size(FRAMEW, FRAMEH));// for saving frame
//
//		VideoWriter writer_result_Blink(
//			result_BlinkWriteOutFile
//			, CV_FOURCC('M', 'J', 'P', 'G'), 15.f, Size(FRAMEW, FRAMEH));// for saving frame
//
//		VideoWriter writer_result_Gaze(
//			result_GazeWriteOutFile
//			, CV_FOURCC('M', 'J', 'P', 'G'), 15.f, Size(calUsedW, calUsedH));// for saving frame
//#endif
			//VideoWriter writer_result_Blink(
			//result_BlinkWriteOutFile
			//, CV_FOURCC('M', 'J', 'P', 'G'), 15.f, Size(FRAMEW, FRAMEH));// for saving frame

			//VideoWriter writer_result_wh(
			//result_WhiteBalanceWriteOutFile
			//, CV_FOURCC('M', 'J', 'P', 'G'), 15.f, Size(FRAMEW, FRAMEH));// for saving frame

			//VideoWriter writer_result_coarseCenter(
			//result_CoarseCenterWriteOutFile
			//, CV_FOURCC('M', 'J', 'P', 'G'), 15.f, Size(FRAMEW, FRAMEH));// for saving frame


			VideoWriter writer_result_EyePos(
						result_EyePosWriteOutFile
						, VideoWriter::fourcc('M', 'J', 'P', 'G'), 15.f, Size(FRAMEW, FRAMEH));// for saving frame

			VideoWriter writer_result_input(
						result_InputFrameOutFile
						, VideoWriter::fourcc('M', 'J', 'P', 'G'), 15.f, Size(FRAMEW, FRAMEH));// for saving frame

			VideoWriter writer_result_wh(
						result_WhiteBalanceWriteOutFile
						, VideoWriter::fourcc('M', 'J', 'P', 'G'), 15.f, Size(FRAMEW, FRAMEH));// for saving frame

			VideoWriter writer_result_ft(
						result_Fitting
						, VideoWriter::fourcc('M', 'J', 'P', 'G'), 15.f, Size(FRAMEW, FRAMEH));// for saving frame

			VideoWriter writer_result_roi(
						result_ROI
						, VideoWriter::fourcc('M', 'J', 'P', 'G'), 15.f, Size(FRAMEW, FRAMEH));// for saving frame

		//===========Check Camera Set==========/
        if(!cap.isOpened()){
                cout << "Error"; getchar();return -1;
        }

		if(!cap_scene.isOpened()){
			cout << "Error"; getchar();return -1;
		}


					
		//===========Algorithm Begin Here==========/		 
		int frame_number = 0;	
		int frame_number_outputFps = 0;
		double avg_fps = 0;
		int countForColorModelHistValidTesting = 0;			
		Point leftEyeCorner, rightEyeCorner;			
		iris_maskModel_refreshFrame = iris_maskModel_refreshFrame_Initial;

		/*Set thread pool with thread_num threads using TBB*/
		tbb::task_scheduler_init init(thread_num);


		/*Test computation time of each function*/
		sprintf(result_Profiling, "%s\\%s\\time_compute_analysis.txt" , testVariancePtsDir, testVarianceFolder);
		ofstream file_time_fpsOut(result_Profiling);
		/*Test computation time of each function*/


		sprintf(IrisHisto_output_fileName ,".\\txt\\IrisHistoModel3D.txt");	
		while(cap.read(Frame) && cap_scene.read(Scene_image)){

			//===========Camera Distortion Free==========/			
			Mat temp = Frame.clone();

            undistort(temp, Frame, CameraMatrix_Eye, DistCoeffs_Eye);


			//PP-s19 final projet do not use scene image
			//if(!(eyeGazeTestProcedure || calibrationProcedureBegin)){
			//	Mat temp = Scene_image.clone();
			//	undistort(temp, Scene_image, CameraMatrix_Scene, DistCoeffs_Scene);
			//}

			//===========Mat Scene_image Resize==========//				
			if(calibrationProcedureDone && !eyeGazeTestProcedure){
				if(changeScene==0){	
					Scene_image = Scene_chessboard.clone();
				}else if(changeScene==1){
					Scene_image = imread(imageTestScene, cv::IMREAD_COLOR);
				}else if(changeScene==2){
					Scene_image = imread(imageTestScene_words, cv::IMREAD_COLOR);
				}	
			}
			resize(Scene_image , Scene_image , Size(calUsedW , calUsedH) , 0 , 0 , INTER_NEAREST);		
			//============White Balence============/
			//Component_Stretching(Frame , Frame_wh);	
			Parallel_Component_Stretching(Frame, Frame_wh, thread_num);

		
			//===========Check Whether Already Set Eye Corner==========/			
			if(!setEyeCornerAndEyePosReady && !doNotdisplayEyeCorner){
				Draw_Cross(Frame_wh, leftCornerOriginalPoint.x, leftCornerOriginalPoint.y, 20, 20, Scalar(255 , 255 , 50) , 3);		
				Draw_Cross(Frame_wh, rightCornerOriginalPoint.x, rightCornerOriginalPoint.y, 20, 20, Scalar(255 , 255 , 50) , 3);		

			
				if(!(RefineEyeCornerROI( roi_LeftWidth , roi_LeftHeight ,roi_RightWidth,roi_RightHeight
													 , rectLeftCornerStartPoint, rectRightCornerStartPoint
													 , leftCornerOriginalPoint , rightCornerOriginalPoint
													 , leftCorrect_x_left , leftCorrect_x_right , leftCorrect_y_up , leftCorrect_y_down
													 , rightCorrect_x_left ,rightCorrect_x_right , rightCorrect_y_up , rightCorrect_y_down
													 , width_LeftROIforFollowingTM ,height_LeftROIforFollowingTM
													 , width_RightROIforFollowingTM ,height_RightROIforFollowingTM)))return 1;			

				rectangle(Frame_wh, rectLeftCornerStartPoint, rectLeftCornerStartPoint+Point(width_LeftROIforFollowingTM , height_LeftROIforFollowingTM)
						, Scalar(0,200 , 255) , 3);

				rectangle(Frame_wh, rectRightCornerStartPoint, rectRightCornerStartPoint+Point(width_RightROIforFollowingTM , height_RightROIforFollowingTM)
						, Scalar(0,200 , 255) , 3);
			}			

			if (do_profiling==false) {//Use the original flow to calculate avg_fps
				t = (double)getTickCount();
			}

			if(setEyeCornerAndEyePosReady){		
				//Frame_Gray = Mat::zeros(FRAMEH, FRAMEW, CV_8UC1);
				//============Convert Color Space============/
				time_start = getTickCount();

				//cvtColor(Frame_wh , Frame_Gray , COLOR_BGRA2GRAY);
				//cv::parallel_for_(cv::Range(0, thread_num), Parallel_cvtColor(Frame_wh, Frame_Gray, COLOR_BGRA2GRAY, thread_num));
				cv::parallel_for_(cv::Range(0, thread_num), Parallel_cvtColor_my_ver(Frame_wh, Frame_Gray, thread_num));

				time_end = getTickCount();
				time_image_preprocessing_cvt_color.push_back(time_end - time_start);

				//============Histogram Equalization============/				
				time_start = getTickCount();

				//equalizeHist(Frame_Gray, Histogram_Eq);
				//std::cout << "0" << std::endl;
				ParallelHistEqual(Frame_Gray, Histogram_Eq, thread_num);

				time_end = getTickCount();
				time_image_preprocessing_hist_equal.push_back(time_end - time_start);

				//imshow("Frame_Gray", Frame_Gray);
				//imshow("Histogram_Eq", Histogram_Eq);
				//waitKey(0);
				//============Iris Model & Mask Generation============/	
				if (countForColorModelHistValidTesting % 5 == 0) {
					EyeImageForTestingIrisHistModel = Frame_wh.clone();
				}

				IrisModelHandeling(gotIrisROI , irisDynamicMaskGeneration , readIrisModel
											  , writeIrisModel , IrisHisto_output_fileName 
											  , Iris_ROI_forModel  ,  Frame_wh , Iris_hist , Iris_Mask
											  , channels , Iris_histSize , ranges , Iris_PerBin , Iris_hist_vector
											  , EyeImageForTestingIrisHistModel
											  , IrisRegionValidTesting , caculateIris_Mask_done
											  , countRefreshTimes);
				
				if(caculateIris_Mask_done && countRefreshTimes==1){
					iris_maskModel_refreshFrame = iris_maskModel_refreshFrame_AfterGet;//Not to find hist model hard
				}

				
				//============Eye Position Detection============/					
				bool bigmotionIrisCenter = false;
				bool noLimbusFeaturePts = false;
				bool getIrisContourPoints = false;
				bool extremeRight_forBlink = false;
				bool extremeUp_forBlink = false;
				bool extremeDown_forBlink = false;	
				bool extremeLeft_forBlink = false;
				bool caculateComplete = false;
				float irisContour_size = 0;				
				Point vertexParabolaLower;
				Point vertexParabolaUpper;
				vector<Point> IrisContoursPoints;
				eyeRefinedIrisCenter = Point(0 , 0);
				EyePosition_Result = Mat::zeros(Frame_wh.size() , CV_8UC3);				
				EyePosition_CenterResult = Frame_wh.clone();
				
				EyePositionDetection(frame_number , Frame , Histogram_Eq 
												, EyePosition_Result , EyePosition_CenterResult , irisCenterEstimation 												
												, Iris_Mask , caculateIris_Mask_done , IrisRegionValidTesting
												, countForColorModelHistValidTesting ,bigmotionIrisCenter , noLimbusFeaturePts
												, vertexParabolaLower , vertexParabolaUpper
												, vertexParabolaUpperFirstFrame , vertexParabolaLowerFirstFrame
												, IrisContoursPoints , getIrisContourPoints
												, extremeRight_forBlink , extremeDown_forBlink 
												, extremeUp_forBlink  , extremeLeft_forBlink
												, irisContour_size
												, eyeCloseDetermine_irisContourSizeThreshold_colorModelBased
												, eyeCoarseIrisCenter_ExtremeRegion_Right , eyeCoarseIrisCenter_ExtremeRegion_Up
												, eyeCoarseIrisCenter_ExtremeRegion_Down, eyeCoarseIrisCenter_ExtremeRegion_Left
												, eyeCenter_rightestPos , eyeCenter_lowestPos
												, eyeCenter_uppestPos , eyeCenter_leftestPos
												, iris_boundaty_Left , iris_boundaty_Right
												, eyeRefinedIrisCenter , caculateComplete
												, calibrationProcedureBegin
												, posNonlinearRegionEyeQueueY
												, file_time_fpsOut
												, writer_result_roi);

				//============Eye Blinking Detection============/			
				if (turn_on_eye_blink_detection) {
					EyeBlinkDetection(noLimbusFeaturePts, caculateIris_Mask_done
						, pre_eyeState, eyeState
						, voluntaryBlinkCount, non_voluntaryBlinkCount
						, countCloseFrame, IrisContoursPoints, getIrisContourPoints
						, extremeRight_forBlink, extremeDown_forBlink
						, extremeUp_forBlink, extremeLeft_forBlink
						, ratio_queue, irisContour_size
						, eyeCloseDetermine_irisContourSizeThreshold_colorModelBased);
				}
			

				//---------------Test Variance can be removed finally---------------//
				if(caculateComplete){
					++countForColorModelHistValidTesting;
					centerEstimation_convexHullFineCenter.push_back(eyeRefinedIrisCenter);
				}
			}
			//============Gaze Estimation============/
			//if (printDebug) {
			//	printf("\n 13\n");
			//}
			if(calibrationProcedureDone){
				if(eyeState==Eyeopen){

					switch(calBrationMethod){
						case calibrationMethod::BilinearInterpolation:{
									GazeEstimation_BilinearInterpolation(gazePoint , eyeRefinedIrisCenter 
																							, calibratedEyeRefinedCenter
																							, calibratedCalPoints
																							, mappingSliceMap
																							, isChessBoardSideOdd);
									}
									break;
						case calibrationMethod::HomographySliceMapping:{
									GazeEstimation_HomographySliceMapping(gazePoint , eyeRefinedIrisCenter																			
																									, mappingSliceMap , calibratedCalPoints);
												  
									}
									break;
						case calibrationMethod::Polynomial_Order:{									
									GazeEstimation_PolyNomial(gazePoint , eyeRefinedIrisCenter 
																				, EyePtsTransformMat_Opt , ScenePtsTransformMat_Opt
																				, calibrationInterMappingFunctionOrder 
																				, mapping_paramOptX , mapping_paramOptY);	
									}
									break;
						case calibrationMethod::Polynomial_RANSAC:{
									GazeEstimation_PolyNomial(gazePoint , eyeRefinedIrisCenter 
																				, EyePtsTransformMat_Opt , ScenePtsTransformMat_Opt
																				, calibrationInterMappingFunctionOrder 
																				, mapping_paramOptX , mapping_paramOptY);												  
									}
									break;
						case calibrationMethod::Polynomial_All:{
									GazeEstimation_PolyNomial(gazePoint , eyeRefinedIrisCenter 
																				, EyePtsTransformMat_Opt , ScenePtsTransformMat_Opt
																				, calibrationInterMappingFunctionOrder 
																				, mapping_paramOptX , mapping_paramOptY);											  
									}
									break;
						case calibrationMethod::SupportVectorRegression:{
												  
									}
									break;
					}
					if(!eyeGazeTestProcedure){
						line(Scene_image , Point(0 , gazePoint.y) , Point(Scene_image.cols - 1 , gazePoint.y) , Scalar(150 , 200 , 220) , 2);
						line(Scene_image , Point(gazePoint.x , 0) , Point(gazePoint.x , Scene_image.rows - 1) , Scalar(150 , 200 , 220) , 2);
					}
				}			
			}
			//if (printDebug) {
			//	printf("\n 14\n");
			//}
			//============Eye Gaze Testing Procedure============/		
			float sqrtTestPtsNum = sqrtf(testNumOfPts);
			int numXTestPts = ceil(sqrtTestPtsNum);
			int numYTestPts = ceil(testNumOfPts/(float)numXTestPts);
			int x_distTest = testUsedW/(numXTestPts);
			int y_distTest = testUsedH/(numYTestPts);
			if(calibrationProcedureDone && eyeGazeTestProcedure){				
				EyeGazeTesting(countInEyeGazeTestingFunction , testPtsVec , Scene_gazetest
										, x_distTest , y_distTest , rand_numTestPts 
										, time_duringTest , time_duringCalLast
										, numXTestPts , numYTestPts
										, gazeTesting_vec
										, gazePoint
										, dispTestPts , dispTestPtsLast
										, setEyeCornerAndEyePosReady
										, eyeState
										, eyeGazeTestProcedure
										, Scene_image
										, countGazePut
										, realGazeInput);	
				Scene_image = Scene_gazetest.clone();		
				if(realGazeInput){
					line(Scene_image , Point(0 , gazePoint.y) , Point(Scene_image.cols - 1 , gazePoint.y) , Scalar(150 , 200 , 220) , 2);
					line(Scene_image , Point(gazePoint.x , 0) , Point(gazePoint.x , Scene_image.rows - 1) , Scalar(150 , 200 , 220) , 2);		
				}
			}
			
			//if (printDebug) {
			//	printf("\n 15\n");
			//}
			//============Calibration Procedure============/					
			if(calibrationProcedureBegin){		
				int numXSpace = calPtsLength;
				int numYSpace = numXSpace;
				int x_dist = calUsedW/numXSpace;
				int y_dist = calUsedH/numYSpace;
				bool calibrationProcedureWholeDone = false;
				calibrationProcedureDone = false;						
				calibrationProcedureWholeDone = CalibrationProcedureWhole(Scene_chessboard , Scene_calibration , count_calProcedure , time_cal
																											, time_duringLast , count_times_2secs , posLine_y
																											, x_dist , y_dist 
																											, getCalibrationPoints , setEyeCornerAndEyePosReady
																											, test_count_cal , calibrationProcedureBegin
																											, eyeRefinedIrisCenter
																											, calibratedEyeRefinedCenter , calibratedCalPoints
																											, posQueueX , posQueueY
																											, filteredEyeRefinedCenter
																											, eyeState																											
																											, calibratedChessBoardPtsLast
																											, calInterAnchorPoints
																											, calBrationMethod
																											, count_times_2secsNext
																											, isChessBoardSideOdd
																											, numXSpace , numYSpace
																											, countEyePut																											
																											, calibrationPatternPtsStep_One
																											, countCalibrationPatternPtsStep_One_pos);
				//Computing Mapping Model
				if(!calibrationProcedureBegin && calibrationProcedureWholeDone){					
					mappingSliceMap.clear();
					printf("\ncaculate mapping function . . . \n");

					//Build Slicing Map
					if(calibrationPts_space==calibrationPattern::Step_space_two && 
						(calBrationMethod==calibrationMethod::BilinearInterpolation || calBrationMethod==calibrationMethod::HomographySliceMapping)){
						getSlicingMap = BuildSlicingMap(calInterAnchorPoints 
																		, calibratedEyeRefinedCenter , calibratedCalPoints
																		, numXSpace , numYSpace
																		, x_dist , y_dist
																		, mappingSliceMap
																		, calBrationMethod
																		, isChessBoardSideOdd);
					}		

					switch(calBrationMethod){
						case calibrationMethod::BilinearInterpolation:{
									calibrationProcedureDone = true;
									}
									break;
						case calibrationMethod::HomographySliceMapping:{
									calibrationProcedureDone = MappingEyeGaze_HomographySlice(mappingSliceMap	, calibratedCalPoints);										
									}
									break;
						case calibrationMethod::Polynomial_Order:{										
									calibrationProcedureDone = MappingEyeGaze_PolyNomialALLPtsCaculated(calibratedEyeRefinedCenter 
																																						, calibratedCalPoints
																																						, mapping_paramOptX , mapping_paramOptY 
																																						, calibrationInterMappingFunctionOrder
																																						, numberOfVar
																																						, EyePtsTransformMat_Opt
																																						, ScenePtsTransformMat_Opt
																																						, meanSquareError);	
									}
									break;
						case calibrationMethod::Polynomial_RANSAC:{
									calibrationProcedureDone = MappingEyeGaze_PolyNomialRANSAC(calibratedEyeRefinedCenter 
																																		, calibratedCalPoints
																																		, mapping_paramOptX , mapping_paramOptY 
																																		, calibrationInterMappingFunctionOrder
																																		, numberOfVar
																																		, EyePtsTransformMat_Opt
																																		, ScenePtsTransformMat_Opt);																					  
									}
									break;
						case calibrationMethod::Polynomial_All:{
									calibrationProcedureDone = MappingEyeGaze_PolyNomialALLOrderCaculated(calibratedEyeRefinedCenter 
																																						, calibratedCalPoints
																																						, mapping_paramOptX , mapping_paramOptY 
																																						, numberOfVar
																																						, EyePtsTransformMat_Opt
																																						, ScenePtsTransformMat_Opt
																																						, calPtsLength
																																						, orderOpt
																																						, calibrationPts_space);	
									calibrationInterMappingFunctionOrder = orderOpt;															 
									}
									break;
						case calibrationMethod::SupportVectorRegression:{
												  
									}
									break;
					}

					
					if(calibrationProcedureDone){	
						printf("\ncalibration done . . .\n");
					}else{
						printf("\ncalibration failed . . .\n");
					}
					count_calProcedure = 0;
				}

				if(!calibrationProcedureWholeDone){
					calibrationProcedureBegin = false;			
					printf("\ncalibration failed . . .\n");
					count_calProcedure = 0;
				}
			}


			if (do_profiling) {
				t = (double)(                                                   +
				(time_image_preprocessing_white_balance.size()>0? time_image_preprocessing_white_balance.back():0)                            +
				(time_image_preprocessing_cvt_color.size()    >0? time_image_preprocessing_cvt_color.back():0)                                +
				(time_image_preprocessing_hist_equal.size()   >0? time_image_preprocessing_hist_equal.back():0)                               +
				(time_image_preprocessing_high_freq_removal.size()>0 ? time_image_preprocessing_high_freq_removal.back() : 0)                 +
				(caculateIris_Mask_done? (time_eye_position_detection_coarse_iris_location_color.size()>0? time_eye_position_detection_coarse_iris_location_color.back():0) : time_eye_position_detection_coarse_iris_location_cdf.size()>0? time_eye_position_detection_coarse_iris_location_cdf.back():0) + 
				(time_eye_position_detection_iris_region_extraction.size()>0 ? time_eye_position_detection_iris_region_extraction.back() : 0) +
				(time_iris_model_calculate_h_s_histogram.size()   >0 ? time_iris_model_calculate_h_s_histogram.back():0)                      +
				(time_iris_model_valid_testing.size()   >0 ? time_iris_model_valid_testing.back() : 0)                                        +
				(time_eye_position_detection_eyelid_feature_detection.size()>0?time_eye_position_detection_eyelid_feature_detection.back():0) +
				(time_eye_position_detection_parabola_model_fitting.size()>0?time_eye_position_detection_parabola_model_fitting.back():0)     +
				(time_eye_position_detection_limbus_feature_detection.size()>0?time_eye_position_detection_limbus_feature_detection.back():0) +
				(time_eye_position_detection_center_calculation.size()>0?time_eye_position_detection_center_calculation.back():0)             +
				(time_eye_position_detection_refresh_min_area_region.size()>0? time_eye_position_detection_refresh_min_area_region.back():0)
				) / getTickFrequency();
			}else {
				t = (double)(getTickCount() - t +
					(time_image_preprocessing_white_balance.size()>0? time_image_preprocessing_white_balance.back():0)
					) / getTickFrequency();
			}

			++frame_number;	

			if(outputAvgFps){
				avg_fps += t;		
				++frame_number_outputFps;
			}

			//if (printDebug) {
			//	printf("\n 17\n");
			//}
			//============Set Output GUI============/		
			char s[300];
			char s2[300];
			sprintf(s , "Frame rate = %lf fps" ,1.f/t);
			
			putText(EyePosition_Result
						, s
						, Point(50 , 50)
						, FONT_HERSHEY_PLAIN
						, 1.5/480*Frame.cols
						, Scalar(0, 255 , 255)
						, 4
						,8
						, false );

			sprintf(s2 , "frame_number = %d" ,frame_number);
			putText(EyePosition_Result
						, s2
						, Point(75 , 75)
						, FONT_HERSHEY_PLAIN
						, 1.5/480.0*Frame.cols
						, Scalar(0, 255 , 255)
						, 4
						,8
						, false );

			sprintf(s , "Voluntary = %d" ,voluntaryBlinkCount);
			putText(EyePosition_Result
						, s
						, Point(Frame.cols - 400 , Frame.rows - 25)
						, FONT_HERSHEY_PLAIN
						, 1.5/480.0*Frame.cols
						, Scalar(255, 255 , 10)
						, 4
						,8
						, false );
			
			if(showDetail){
				imshow("Input_Frame",Frame);
				imshow("White_Balence", Frame_wh);
				//imshow("Scene",Scene_image);											
			}

			if(calibrationProcedureBegin){
				imshow("Scene" , Scene_calibration);
		/*	}else if(eyeGazeTestProcedure){
				imshow("Scene" , Scene_gazetest);*/
			}else{
				//imshow("Scene" , Scene_image);
			}

			if(setEyeCornerAndEyePosReady){
				//imshow("Gray", Frame_Gray);
				//imshow("Histogram_Eq" , Histogram_Eq);								
				//imshow("Parabola" , Parabola);	
				imshow("EyePosition_Result" , EyePosition_Result);
				imshow("EyePosition_CenterResult" , EyePosition_CenterResult);			
			}

			char output_img_name[MAX_WORD_LEN];
			char key = (char)waitKey(1);			
			if(key==32/*Space*/){
				printf("Space\n");
				waitKey(0);
			}else if(key==27){
				break;			
			}else if(key=='r'){				
				printf("r");
				enterstring.push_back(key);
			}else if(key=='d'){					
				printf("d");
				enterstring.push_back(key);
			}else if(key=='s'){								
				printf("s");
				enterstring.push_back(key);
			}else if(key=='w'){								
				printf("w");
				enterstring.push_back(key);
			}else if(key=='.'){		
				printf(".");
				enterstring.push_back(key);
			}else if(key=='c'){
				printf("c");
				enterstring.push_back(key);
			}else if(key=='a'){
				printf("a");
				enterstring.push_back(key);
			}else if(key=='l'){
				printf("l");
				enterstring.push_back(key);
			}else if(key=='p'){	
				printf("p");
				enterstring.push_back(key);
			}else if(key=='o'){
				printf("o");				
				enterstring.push_back(key);
			}else if(key=='t'){
				printf("t");
				enterstring.push_back(key);
			}else if(key=='e'){
				printf("e");
				enterstring.push_back(key);
			}else if(key=='i'){
				printf("i");
				enterstring.push_back(key);
			}else if(key=='_'){
				printf("_");
				enterstring.push_back(key);
			}else if(key=='f'){
				printf("f");
				enterstring.push_back(key);
			}else if(key=='b'){
				printf("b");
				enterstring.push_back(key);
			}else if(key=='n') {
				printf("n");
				enterstring.push_back(key);
			}else if(key=='k') {
				printf("k");
				enterstring.push_back(key);
			}else if(key==13/*Enter*/){				
				if(enterstring=="cal"){
					calibrationProcedureBegin = true;
					printf("\ncalibration begin . . . \n");	
				}else if(enterstring=="d"){	
					doNotdisplayEyeCorner = !doNotdisplayEyeCorner;
				}else if(enterstring=="s"){
					setEyeCornerAndEyePosReady = !setEyeCornerAndEyePosReady;
				}else if(enterstring=="w"){
					writeIrisModel = true;					
				}else if(enterstring=="p"){
					++changeScene;
					changeScene = changeScene%4;				
				}else if(enterstring=="test"){
					eyeGazeTestProcedure = true;
				}else if(enterstring=="test_pts"){					
					printf("\n How many test points ? \n");
					scanf("%d" , &testNumOfPts);
				}else if(enterstring=="cal_len"){					
					printf("\n Calibration Length NxN ? \n");
					scanf("%d" , &calPtsLength);
				}else if(enterstring=="test_fps"){
					outputAvgFps = !outputAvgFps;
					if(outputAvgFps){
						printf("\n outputFps = true\n");
					}else{
						printf("\n outputFps = false\n");
					}
				}else if (enterstring == "blink") {
					turn_on_eye_blink_detection = true;
				}else if (enterstring == "profile") {
					do_profiling = true;
				}

				enterstring.clear();
				printf("\n");
			}


			//if (printDebug) {
			//	printf("\n 18\n");
			//}
			//============Save Input Frame============/		
			if (setEyeCornerAndEyePosReady){
				writer_result_EyePos.write(EyePosition_CenterResult);
				writer_result_ft.write(EyePosition_Result);
				writer_result_input.write(Frame);
				writer_result_wh.write(Frame_wh);
			}

//#if writeResult
//				writer_result_Gaze.write(Scene_image);
//
//				writer_result_EyePos.write(EyePosition_CenterResult);
//		
//				writer_result_Blink.write(EyePosition_Result);
//
//				if(calibrationProcedureBegin){
//					writer_Cal.write(Scene_calibration);
//				}
//				if(eyeGazeTestProcedure){
//					writer_GazeTesting.write(Scene_image);
//				}
//#endif
				//if(getEyeCoarseCenterGlob){
				//	writer_result_Blink.write(EyePosition_Result);
				//	writer_result_wh.write(Frame_wh);
				//	writer_result_coarseCenter.write(CoarseCenterDispLightInference);
				//}
			//============Release Daata============/
			//waitKey(0);
			//Frame.release();			
			//Frame_Gray.release();
			//Histogram_Eq.release();
			//Frame_wh.release();	
			//EyePosition_CenterResult.release();
			//EyePosition_Result.release();
			//Parabola.release();				

			//Frame.u->refcount = 0;
			//Frame_Gray.u->refcount = 0;
			//Histogram_Eq.u->refcount = 0;
			//Frame_wh.u->refcount = 0;
			//EyePosition_Result.refcount = 0;
			//Parabola.refcount = 0;	

			//if (printDebug) {
			//	printf("\n 19\n");
			//}
        }//end while


		//if (printDebug) {
		//	printf("\n 20\n");
		//}
		//======Testing Variance======/
		if(testVariance){
			sprintf(testVarOutFileName, "%s\\%s", testVariancePtsDir, testVarianceFolder);
			WriteOutTestingData(testVarOutFileName);
		}

		//======Profiling======/
		if (do_profiling) {
			if (time_image_preprocessing_white_balance.size() > 0) {
				avg_time = std::accumulate(time_image_preprocessing_white_balance.begin(), time_image_preprocessing_white_balance.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_image_preprocessing_white_balance.size();
				file_time_fpsOut << "time_image_preprocessing_white_balance      = " << avg_time << endl;
			}
			if (time_image_preprocessing_cvt_color.size() > 0) {
				avg_time = std::accumulate(time_image_preprocessing_cvt_color.begin(), time_image_preprocessing_cvt_color.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_image_preprocessing_cvt_color.size();
				file_time_fpsOut << "time_image_preprocessing_cvt_color          = " << avg_time << endl;
			}
			if (time_image_preprocessing_hist_equal.size() > 0) {
				avg_time = std::accumulate(time_image_preprocessing_hist_equal.begin(), time_image_preprocessing_hist_equal.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_image_preprocessing_hist_equal.size();
				file_time_fpsOut << "time_image_preprocessing_hist_equal         = " << avg_time << endl;
			}
			if (time_image_preprocessing_high_freq_removal.size() > 0) {
				avg_time = std::accumulate(time_image_preprocessing_high_freq_removal.begin(), time_image_preprocessing_high_freq_removal.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_image_preprocessing_high_freq_removal.size();
				file_time_fpsOut << "time_image_preprocessing_high_freq_removal  = " << avg_time << endl;
			}
			if (time_eye_position_detection_coarse_iris_location_cdf.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_coarse_iris_location_cdf.begin(), time_eye_position_detection_coarse_iris_location_cdf.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_coarse_iris_location_cdf.size();
				file_time_fpsOut << "time_eye_position_detection_coarse_iris_location_cdf          = " << avg_time << endl;
			}
			if (time_eye_position_detection_coarse_iris_location_color.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_coarse_iris_location_color.begin(), time_eye_position_detection_coarse_iris_location_color.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_coarse_iris_location_color.size();
				file_time_fpsOut << "time_eye_position_detection_coarse_iris_location_color        = " << avg_time << endl;
			}
			if (time_eye_position_detection_coarse_iris_location_cdf_serial.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_coarse_iris_location_cdf_serial.begin(), time_eye_position_detection_coarse_iris_location_cdf_serial.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_coarse_iris_location_cdf_serial.size();
				file_time_fpsOut << "time_eye_position_detection_coarse_iris_location_cdf_serial   = " << avg_time << endl;
			}
			if (time_eye_position_detection_coarse_iris_location_color_serial.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_coarse_iris_location_color_serial.begin(), time_eye_position_detection_coarse_iris_location_color_serial.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_coarse_iris_location_color_serial.size();
				file_time_fpsOut << "time_eye_position_detection_coarse_iris_location_color_serial = " << avg_time << endl;
			}
			if (time_eye_position_detection_iris_region_extraction.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_iris_region_extraction.begin(), time_eye_position_detection_iris_region_extraction.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_iris_region_extraction.size();
				file_time_fpsOut << "time_eye_position_detection_iris_region_extraction            = " << avg_time << endl;
			}
			if (time_eye_position_detection_eyelid_feature_detection.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection.begin(), time_eye_position_detection_eyelid_feature_detection.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection.size();
				file_time_fpsOut << "time_eye_position_detection_eyelid_feature_detection          = " << avg_time << endl;
			}
			if (time_eye_position_detection_eyelid_feature_detection_district1.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district1.begin(), time_eye_position_detection_eyelid_feature_detection_district1.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district1.size();
				file_time_fpsOut << ">>>> time_eye_position_detection_eyelid_feature_detection_district1   = " << avg_time << endl;
			}
			//if (time_eye_position_detection_eyelid_feature_detection_district1_gaussianblur.size() > 0) {
			//	avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district1_gaussianblur.begin(), time_eye_position_detection_eyelid_feature_detection_district1_gaussianblur.end(), 0);
			//	avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district1_gaussianblur.size();
			//	file_time_fpsOut << ">>>>>> time_eye_position_detection_eyelid_feature_detection_district1_gaussianblur   = " << avg_time << endl;
			//}
			//if (time_eye_position_detection_eyelid_feature_detection_district1_gaussianblur_core.size() > 0) {
			//	avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district1_gaussianblur_core.begin(), time_eye_position_detection_eyelid_feature_detection_district1_gaussianblur_core.end(), 0);
			//	avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district1_gaussianblur_core.size();
			//	file_time_fpsOut << ">>>>>> time_eye_position_detection_eyelid_feature_detection_district1_gaussianblur_core   = " << avg_time << endl;
			//}
			if (time_eye_position_detection_eyelid_feature_detection_district2.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district2.begin(), time_eye_position_detection_eyelid_feature_detection_district2.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district2.size();
				file_time_fpsOut << ">>>> time_eye_position_detection_eyelid_feature_detection_district2   = " << avg_time << endl;
			}
			if (time_eye_position_detection_eyelid_feature_detection_district3.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district3.begin(), time_eye_position_detection_eyelid_feature_detection_district3.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district3.size();
				file_time_fpsOut << ">>>> time_eye_position_detection_eyelid_feature_detection_district3   = " << avg_time << endl;
			}
			if (time_eye_position_detection_eyelid_feature_detection_district4.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district4.begin(), time_eye_position_detection_eyelid_feature_detection_district4.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district4.size();
				file_time_fpsOut << ">>>> time_eye_position_detection_eyelid_feature_detection_district4   = " << avg_time << endl;
			}
			if (time_eye_position_detection_eyelid_feature_detection_district4_0.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district4_0.begin(), time_eye_position_detection_eyelid_feature_detection_district4_0.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district4_0.size();
				file_time_fpsOut << ">>>>>> time_eye_position_detection_eyelid_feature_detection_district4_0           = " << avg_time << endl;
			}
			if (time_eye_position_detection_eyelid_feature_detection_district4_1.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district4_1.begin(), time_eye_position_detection_eyelid_feature_detection_district4_1.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district4_1.size();
				file_time_fpsOut << ">>>>>> time_eye_position_detection_eyelid_feature_detection_district4_1           = " << avg_time << endl;
			}
			if (time_eye_position_detection_eyelid_feature_detection_district4_1_1.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district4_1_1.begin(), time_eye_position_detection_eyelid_feature_detection_district4_1_1.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district4_1_1.size();
				file_time_fpsOut << ">>>>>>>> time_eye_position_detection_eyelid_feature_detection_district4_1_1           = " << avg_time << endl;
			}
			if (time_eye_position_detection_eyelid_feature_detection_district4_1_2.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district4_1_2.begin(), time_eye_position_detection_eyelid_feature_detection_district4_1_2.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district4_1_2.size();
				file_time_fpsOut << ">>>>>>>> time_eye_position_detection_eyelid_feature_detection_district4_1_2           = " << avg_time << endl;
			}
			if (time_eye_position_detection_eyelid_feature_detection_district4_1_3.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district4_1_3.begin(), time_eye_position_detection_eyelid_feature_detection_district4_1_3.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district4_1_3.size();
				file_time_fpsOut << ">>>>>>>> time_eye_position_detection_eyelid_feature_detection_district4_1_3           = " << avg_time << endl;
			}
			if (time_eye_position_detection_eyelid_feature_detection_district4_1_4.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district4_1_4.begin(), time_eye_position_detection_eyelid_feature_detection_district4_1_4.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district4_1_4.size();
				file_time_fpsOut << ">>>>>>>> time_eye_position_detection_eyelid_feature_detection_district4_1_4           = " << avg_time << endl;
			}
			if (time_eye_position_detection_eyelid_feature_detection_district4_1_5.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district4_1_5.begin(), time_eye_position_detection_eyelid_feature_detection_district4_1_5.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district4_1_5.size();
				file_time_fpsOut << ">>>>>>>> time_eye_position_detection_eyelid_feature_detection_district4_1_5           = " << avg_time << endl;
			}
			if (time_eye_position_detection_eyelid_feature_detection_district4_2.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district4_2.begin(), time_eye_position_detection_eyelid_feature_detection_district4_2.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district4_2.size();
				file_time_fpsOut << ">>>>>> time_eye_position_detection_eyelid_feature_detection_district4_2           = " << avg_time << endl;
			}
			//if (time_eye_position_detection_eyelid_feature_detection_district4_2_core.size() > 0) {
			//	avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district4_2_core.begin(), time_eye_position_detection_eyelid_feature_detection_district4_2_core.end(), 0);
			//	avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district4_2_core.size();
			//	file_time_fpsOut << ">>>>>>>> time_eye_position_detection_eyelid_feature_detection_district4_2_core    = " << avg_time << endl;
			//}
			if (time_eye_position_detection_eyelid_feature_detection_district4_3.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district4_3.begin(), time_eye_position_detection_eyelid_feature_detection_district4_3.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district4_3.size();
				file_time_fpsOut << ">>>>>> time_eye_position_detection_eyelid_feature_detection_district4_3           = " << avg_time << endl;
			}
			if (time_eye_position_detection_eyelid_feature_detection_district4_3_1.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district4_3_1.begin(), time_eye_position_detection_eyelid_feature_detection_district4_3_1.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district4_3_1.size();
				file_time_fpsOut << ">>>>>>>> time_eye_position_detection_eyelid_feature_detection_district4_3_1           = " << avg_time << endl;
			}
			if (time_eye_position_detection_eyelid_feature_detection_district4_3_2.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district4_3_2.begin(), time_eye_position_detection_eyelid_feature_detection_district4_3_2.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district4_3_2.size();
				file_time_fpsOut << ">>>>>>>> time_eye_position_detection_eyelid_feature_detection_district4_3_2           = " << avg_time << endl;
			}
			if (time_eye_position_detection_eyelid_feature_detection_district4_3_3.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district4_3_3.begin(), time_eye_position_detection_eyelid_feature_detection_district4_3_3.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district4_3_3.size();
				file_time_fpsOut << ">>>>>>>> time_eye_position_detection_eyelid_feature_detection_district4_3_3           = " << avg_time << endl;
			}
			if (time_eye_position_detection_eyelid_feature_detection_district4_3_4.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district4_3_4.begin(), time_eye_position_detection_eyelid_feature_detection_district4_3_4.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district4_3_4.size();
				file_time_fpsOut << ">>>>>>>> time_eye_position_detection_eyelid_feature_detection_district4_3_4           = " << avg_time << endl;
			}
			if (time_eye_position_detection_eyelid_feature_detection_district4_4.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district4_4.begin(), time_eye_position_detection_eyelid_feature_detection_district4_4.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district4_4.size();
				file_time_fpsOut << ">>>>>> time_eye_position_detection_eyelid_feature_detection_district4_4           = " << avg_time << endl;
			}
			if (time_eye_position_detection_eyelid_feature_detection_district4_5.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district4_5.begin(), time_eye_position_detection_eyelid_feature_detection_district4_5.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district4_5.size();
				file_time_fpsOut << ">>>>>> time_eye_position_detection_eyelid_feature_detection_district4_5           = " << avg_time << endl;
			}
			if (time_eye_position_detection_eyelid_feature_detection_district4_local_sum.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district4_local_sum.begin(), time_eye_position_detection_eyelid_feature_detection_district4_local_sum.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district4_local_sum.size();
				file_time_fpsOut << ">>>>>> time_eye_position_detection_eyelid_feature_detection_district4_local_sum   = " << avg_time << endl;
			}
			if (time_eye_position_detection_eyelid_feature_detection_district5.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district5.begin(), time_eye_position_detection_eyelid_feature_detection_district5.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district5.size();
				file_time_fpsOut << ">>>> time_eye_position_detection_eyelid_feature_detection_district5   = " << avg_time << endl;
			}
			if (time_eye_position_detection_eyelid_feature_detection_district5_core.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district5_core.begin(), time_eye_position_detection_eyelid_feature_detection_district5_core.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district5_core.size();
				file_time_fpsOut << ">>>>>> time_eye_position_detection_eyelid_feature_detection_district5_core    = " << avg_time << endl;
			}
			if (time_eye_position_detection_eyelid_feature_detection_district5_merge.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_eyelid_feature_detection_district5_merge.begin(), time_eye_position_detection_eyelid_feature_detection_district5_merge.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_eyelid_feature_detection_district5_merge.size();
				file_time_fpsOut << ">>>>>> time_eye_position_detection_eyelid_feature_detection_district5_merge   = " << avg_time << endl;
			}
			if (time_eye_position_test.size() > 0) {
				avg_time = std::accumulate(time_eye_position_test.begin(), time_eye_position_test.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_test.size();
				file_time_fpsOut << ">>>> time_eye_position_test                               = " << avg_time << endl;
			}
			if (time_eye_position_detection_parabola_model_fitting.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_parabola_model_fitting.begin(), time_eye_position_detection_parabola_model_fitting.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_parabola_model_fitting.size();
				file_time_fpsOut << "time_eye_position_detection_parabola_model_fitting     = " << avg_time << endl;
			}
			if (time_eye_position_detection_limbus_feature_detection.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_limbus_feature_detection.begin(), time_eye_position_detection_limbus_feature_detection.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_limbus_feature_detection.size();
				file_time_fpsOut << "time_eye_position_detection_limbus_feature_detection   = " << avg_time << endl;
			}
			if (time_eye_position_detection_limbus_feature_detection_serial.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_limbus_feature_detection_serial.begin(), time_eye_position_detection_limbus_feature_detection_serial.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_limbus_feature_detection_serial.size();
				file_time_fpsOut << ">> time_eye_position_detection_limbus_feature_detection_serial   = " << avg_time << endl;
			}
			if (time_eye_position_detection_center_calculation.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_center_calculation.begin(), time_eye_position_detection_center_calculation.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_center_calculation.size();
				file_time_fpsOut << "time_eye_position_detection_center_calculation         = " << avg_time << endl;
			}
			if (time_eye_position_detection_refresh_min_area_region.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_refresh_min_area_region.begin(), time_eye_position_detection_refresh_min_area_region.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_refresh_min_area_region.size();
				file_time_fpsOut << "time_eye_position_detection_refresh_min_area_region    = " << avg_time << endl;
			}
			if (time_eye_position_detection_refresh_iris_ROI_region.size() > 0) {
				avg_time = std::accumulate(time_eye_position_detection_refresh_iris_ROI_region.begin(), time_eye_position_detection_refresh_iris_ROI_region.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_eye_position_detection_refresh_iris_ROI_region.size();
				file_time_fpsOut << ">> time_eye_position_detection_refresh_iris_ROI_region    = " << avg_time << endl;
			}
			if (time_iris_model_calculate_h_s_histogram.size() > 0) {
				avg_time = std::accumulate(time_iris_model_calculate_h_s_histogram.begin(), time_iris_model_calculate_h_s_histogram.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_iris_model_calculate_h_s_histogram.size();
				file_time_fpsOut << "time_iris_model_calculate_h_s_histogram               = " << avg_time << endl;
			}
			if (time_iris_model_valid_testing.size() > 0) {
				avg_time = std::accumulate(time_iris_model_valid_testing.begin(), time_iris_model_valid_testing.end(), 0);
				avg_time = (avg_time / getTickFrequency()) / time_iris_model_valid_testing.size();
				file_time_fpsOut << "time_iris_model_valid_testing                         = " << avg_time << endl;
			}
		}

		//if (printDebug) {
		//	printf("\n 21\n");
		//}
		//======Caculate result======/
		avg_fps/=frame_number_outputFps;
		cout<<"frame_number = "<<frame_number_outputFps<<endl;
		cout<<"average fps = "<<1.f/avg_fps<<endl;
		file_time_fpsOut << "=============================" << endl;
		file_time_fpsOut << "frame_number = " << frame_number << endl;
		file_time_fpsOut << "average fps = "  << 1.f / avg_fps << endl;
		//EstimationError(groundTruth , irisCenterEstimation);

		Frame.release();			
		Frame_Gray.release();
		Histogram_Eq.release();
		Frame_wh.release();	
		EyePosition_CenterResult.release();
		EyePosition_Result.release();
		//if (printDebug) {
		//	printf("\n 22\n");
		//}
		//getchar();
		return 0;
}