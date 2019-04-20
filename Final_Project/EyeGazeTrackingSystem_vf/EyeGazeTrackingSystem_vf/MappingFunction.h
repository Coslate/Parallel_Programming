#ifndef MAPPING_FUNCTION_RANSAC_H
#define MAPPING_FUNCTION_RANSAC_H
#pragma once
#define _USE_MATH_DEFINES 

//#include <dlib\svm.h>
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
#include <utility> 
#include "Parameters.h"
#include "SVD.h"


using namespace cv;
using namespace std;


enum calibrationPattern{
	Step_space_one = 1,
	Step_space_two = 2,
};

enum calibrationMethod{
	BilinearInterpolation ,
	HomographySliceMapping , 
	Polynomial_Order , 
	Polynomial_All , 
	Polynomial_RANSAC ,
	SupportVectorRegression
};

class sliceMapElement{
private:
	pair<Point , int> upAnchorPt;
	pair<Point , int> downAnchorPt;
	pair<Point , int> leftAnchorPt;
	pair<Point , int> rightAnchorPt;

	pair<Point , int> interpolationPt_up;
	pair<Point , int> interpolationPt_down;
	pair<Point , int> interpolationPt_left;
	pair<Point , int> interpolationPt_right;

	pair<float , float>lineEq;
	Mat sliceMap;
	Mat HomoTransMatrix;
	Point2f map_center;
	bool upDelete , downDelete , rightDelete , leftDelete;
	static int testOut;
public:
	sliceMapElement(const Point &up , const Point &down , const Point &left , const Point &right):
								upAnchorPt(pair<Point , int>(up , 0)) , downAnchorPt(pair<Point , int>(down , 0)) 
								, leftAnchorPt(pair<Point , int>(left , 0)) , rightAnchorPt(pair<Point , int>(right , 0))
								, interpolationPt_up(pair<Point , int>(Point(0 , 0) , 0)) , interpolationPt_down(pair<Point , int>(Point(0 , 0) , 0))
								, interpolationPt_left(pair<Point , int>(Point(0 , 0) , 0)) , interpolationPt_right(pair<Point , int>(Point(0 , 0) , 0))
								, upDelete(false) , downDelete(false) , rightDelete(false) , leftDelete(false)
								, lineEq(pair<float , float>(0 , 0)) , map_center(Point2f(0 , 0)){
		sliceMap = Mat::zeros(FRAMEH , FRAMEW , CV_8UC1);
		HomoTransMatrix = Mat::zeros(3 , 3 , CV_64F);
		testOut = 0;
	}

	sliceMapElement(): upAnchorPt(pair<Point , int>(Point(0 , 0) , 0)) , downAnchorPt(pair<Point , int>(Point(0 , 0) , 0)) 
								, leftAnchorPt(pair<Point , int>(Point(0 , 0) , 0)) , rightAnchorPt(pair<Point , int>(Point(0 , 0) , 0))
								, interpolationPt_up(pair<Point , int>(Point(0 , 0) , 0)) , interpolationPt_down(pair<Point , int>(Point(0 , 0) , 0))
								, interpolationPt_left(pair<Point , int>(Point(0 , 0) , 0)) , interpolationPt_right(pair<Point , int>(Point(0 , 0) , 0))
								, upDelete(false) , downDelete(false) , rightDelete(false) , leftDelete(false)
								, lineEq(pair<float , float>(0 , 0)) , map_center(Point2f(0 , 0)){
		sliceMap = Mat::zeros(FRAMEH , FRAMEW , CV_8UC1);
		HomoTransMatrix = Mat::zeros(3 , 3 , CV_64F);
		testOut = 0;
	}

	inline void setAnchorPtsALL(const pair<Point , int> &up , const pair<Point , int> &down 
												, const pair<Point , int> &left , const pair<Point , int> &right){
		upAnchorPt = up;
		downAnchorPt = down;
		leftAnchorPt = left;
		rightAnchorPt = right;
	}

	inline void setAnchorPtsLeftDelete(const pair<Point , int> &up , const pair<Point , int> &down , const pair<Point , int> &right){
		upAnchorPt = up;
		downAnchorPt = down;		
		rightAnchorPt = right;
		leftDelete = true;
	}

	inline void setAnchorPtsRightDelete(const pair<Point , int> &up , const pair<Point , int> &down , const pair<Point , int> &left){
		upAnchorPt = up;
		downAnchorPt = down;		
		leftAnchorPt = left;
		rightDelete = true;
	}

	inline void setAnchorPtsUpDelete(const pair<Point , int> &down , const pair<Point , int> &left , const pair<Point , int> &right){		
		downAnchorPt = down;
		leftAnchorPt = left;
		rightAnchorPt = right;
		upDelete = true;
	}

	inline void setAnchorPtsDownDelete(const pair<Point , int> &up , const pair<Point , int> &left , const pair<Point , int> &right){		
		upAnchorPt = up;
		leftAnchorPt = left;
		rightAnchorPt = right;
		downDelete = true;
	}

	inline void setAnchorPtsUpRightDelete(const pair<Point , int> &down , const pair<Point , int> &left){				
		leftAnchorPt = left;
		downAnchorPt = down;
		upDelete = true;
		rightDelete = true;
	}

	inline void setAnchorPtsDownLeftDelete(const pair<Point , int> &up , const pair<Point , int> &right){				
		rightAnchorPt = right;
		upAnchorPt = up;
		downDelete = true;
		leftDelete = true;
	}

	inline void mapConstruction(){
		vector<Point> contours;	

		if(!(upDelete | downDelete | rightDelete | leftDelete)){			
			contours.push_back(leftAnchorPt.first);
			contours.push_back(upAnchorPt.first);
			contours.push_back(rightAnchorPt.first);
			contours.push_back(downAnchorPt.first);	
		}else{
			if(upDelete & rightDelete){
				Point tmp(downAnchorPt.first.x , leftAnchorPt.first.y);				
				contours.push_back(leftAnchorPt.first);				
				contours.push_back(tmp);
				contours.push_back(downAnchorPt.first);		
			}else if(downDelete & leftDelete){
				Point tmp(upAnchorPt.first.x , rightAnchorPt.first.y);
				contours.push_back(tmp);				
				contours.push_back(upAnchorPt.first);
				contours.push_back(rightAnchorPt.first);	
			}else if(upDelete){
				contours.push_back(leftAnchorPt.first);				
				contours.push_back(rightAnchorPt.first);
				contours.push_back(downAnchorPt.first);	
			}else if(rightDelete){
				contours.push_back(leftAnchorPt.first);				
				contours.push_back(upAnchorPt.first);
				contours.push_back(downAnchorPt.first);	
			}else if(downDelete){
				contours.push_back(leftAnchorPt.first);				
				contours.push_back(upAnchorPt.first);
				contours.push_back(rightAnchorPt.first);	
			}else if(leftDelete){						
				contours.push_back(upAnchorPt.first);
				contours.push_back(rightAnchorPt.first);	
				contours.push_back(downAnchorPt.first);	
			}			
		}
		vector<vector<Point>> hull;
		hull.push_back(contours);
		fillPoly(sliceMap , hull , Scalar::all(255));
	}

	inline void reorganize(){
		if(!(upDelete | downDelete | rightDelete | leftDelete)){			
			interpolationPt_up = upAnchorPt;
			interpolationPt_down = downAnchorPt;
			interpolationPt_left = rightAnchorPt;
			interpolationPt_right = leftAnchorPt;
		}else{
			if(upDelete & rightDelete){
				interpolationPt_right = leftAnchorPt;
				interpolationPt_down = downAnchorPt;
			}else if(downDelete & leftDelete){
				interpolationPt_left = rightAnchorPt;
				interpolationPt_up = upAnchorPt;
			}else if(upDelete){
				interpolationPt_left = rightAnchorPt;
				interpolationPt_right = leftAnchorPt;
				interpolationPt_down = downAnchorPt;
			}else if(rightDelete){
				interpolationPt_right = leftAnchorPt;
				interpolationPt_up = upAnchorPt;
				interpolationPt_down = downAnchorPt;
			}else if(downDelete){
				interpolationPt_left = rightAnchorPt;
				interpolationPt_right = leftAnchorPt;
				interpolationPt_up = upAnchorPt;
			}else if(leftDelete){						
				interpolationPt_left = rightAnchorPt;
				interpolationPt_up = upAnchorPt;
				interpolationPt_down = downAnchorPt;
			}			
		}
	}

	inline void lineEqConstruction(){		
		if(upDelete){
			Point left = interpolationPt_left.first;
			Point right = interpolationPt_right.first;
			lineEq.first = (right.y - left.y)/float(right.x - left.x);
			lineEq.second = (-right.y*left.x + right.x*left.y)/float(right.x - left.x);
		}else if(rightDelete){
			Point left = interpolationPt_up.first;
			Point right = interpolationPt_down.first;
			lineEq.first = (right.y - left.y)/float(right.x - left.x);
			lineEq.second = (-right.y*left.x + right.x*left.y)/float(right.x - left.x);
		}else if(downDelete){
			Point left = interpolationPt_left.first;
			Point right = interpolationPt_right.first;
			lineEq.first = (right.y - left.y)/float(right.x - left.x);
			lineEq.second = (-right.y*left.x + right.x*left.y)/float(right.x - left.x);
		}else if(leftDelete){						
			Point left = interpolationPt_up.first;
			Point right = interpolationPt_down.first;
			lineEq.first = (right.y - left.y)/float(right.x - left.x);
			lineEq.second = (-right.y*left.x + right.x*left.y)/float(right.x - left.x);
		}	
	}

	inline void centerConstruction(){		
		Moments oMoments = moments(sliceMap);

		double dM01 = oMoments.m01;
		double dM10 = oMoments.m10;
		double dArea = oMoments.m00;
		map_center.x = dM10 / dArea;
		map_center.y = dM01 / dArea;  		
	}

	bool mappingHomographyConstruction(const std::vector<Point> &calibratedCalPoints);
	bool mappingHomographyConstructionSelfDefined(const std::vector<Point> &calibratedCalPoints);


	inline const Mat& returnMap() const{
		return sliceMap;
	}
	inline const pair<Point , int>& returnElement(const char dirIndex[]) const{
		if(dirIndex=="up"){
			return upAnchorPt;
		}else if(dirIndex=="down"){
			return downAnchorPt;
		}else if(dirIndex=="left"){
			return leftAnchorPt;
		}else if(dirIndex=="right"){
			return rightAnchorPt;
		}else{
			printf("\nparameter error in returnElement.\n");
		}
	}

	inline const bool& returnDefect(const char dirIndex[]) const{
		if(dirIndex=="up"){
			return upDelete;
		}else if(dirIndex=="down"){
			return downDelete;
		}else if(dirIndex=="left"){
			return leftDelete;
		}else if(dirIndex=="right"){
			return rightDelete;
		}else{
			printf("\nparameter error in returnDefect.\n");
		}
	}

	inline const pair<Point , int>& returnInterpolationPairs(const char dirIndex[]) const{
		if(dirIndex=="up"){
			return interpolationPt_up;
		}else if(dirIndex=="down"){
			return interpolationPt_down;
		}else if(dirIndex=="left"){
			return interpolationPt_left;
		}else if(dirIndex=="right"){
			return interpolationPt_right;
		}else{
			printf("\nparameter error in returnDefect.\n");
		}
	}

	inline const pair<float , float>& returnLineEq() const{
		return lineEq;	
	}

	inline const Point2f& returnCenter() const{
		return map_center;
	}

	inline const Mat& returnHompgraphyMatrix() const{
		return HomoTransMatrix;
	}
};

bool MappingEyeGaze_HomographySlice(std::vector<sliceMapElement> &mappingSliceMap
																		, const std::vector<Point> &calibratedCalPoints);

bool MappingEyeGaze_PolyNomialRANSAC(const std::vector<Point> &calibratedEyeRefinedCenter 
																			, const std::vector<Point> &calibratedCalPoints
																			, double* &mapping_paramX , double* &mapping_paramY
																			, const int &mappingOrder , int &numberOfVar
																			, Mat &EyePtsTransformMat_Opt
																			, Mat &ScenePtsTransformMat_Opt);

bool MappingEyeGaze_PolyNomialALLPtsCaculated(const std::vector<Point> &calibratedEyeRefinedCenter 
																					, const std::vector<Point> &calibratedCalPoints
																					, double* &mapping_paramOptX , double* &mapping_paramOptY 
																					, const int &mappingOrder , int &numberOfVar
																					, Mat &EyePtsTransformMat_Opt
																					, Mat &ScenePtsTransformMat_Opt
																					, double &meanSquareError);

bool MappingEyeGaze_PolyNomialALLOrderCaculated(const std::vector<Point> &calibratedEyeRefinedCenter 
																					, const std::vector<Point> &calibratedCalPoints
																					, double* &mapping_paramOptX , double* &mapping_paramOptY 
																					, int &numberOfVar
																					, Mat &EyePtsTransformMat_Opt
																					, Mat &ScenePtsTransformMat_Opt
																					, const int &calPtsLength
																					, int &orderOpt
																					, int &calibrationPts_space);


//typedef dlib::matrix<double,2,1> sample_type;
//typedef dlib::radial_basis_kernel<sample_type> kernel_type;
//bool MappingEyeGaze_SVR(const std::vector<Point> &calibratedEyeRefinedCenter 
//											, const std::vector<Point> &calibratedCalPoints
//											, dlib::decision_function<kernel_type> &svr_model_X
//											, dlib::decision_function<kernel_type> &svr_model_Y);



//---------------------------------------Eye Gaze Testing Class---------------------------------------//
class GazeTestingElement{
private:
	Point testPts;
	vector<Point> collectGazeEstPts;
public:
	GazeTestingElement(){}
	inline void setTestPts(const Point &inputPts){
		testPts = inputPts;
	}
	inline void setCollectGazePts(const Point &inputPts){
		collectGazeEstPts.push_back(inputPts);
	}

	inline const vector<Point>& returnGazeVec() const{
		return collectGazeEstPts;
	}
	
	inline const Point& returnTestPts() const{
		return testPts;
	}
};



#endif