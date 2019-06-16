#include "Eyelid_FeatureDetection.h"
#include<vector>
#include<cmath>
#include <omp.h>
#include <math.h>

//char thesisImageOutDirEyelid[MAX_WORD_LEN] = "C:\\Users\\sychien\\Desktop\\Oral\\Fig\\EyelidDetection";

inline bool checkpoint(const Point &pt ,  const Mat &src){
	if(pt.x<0||pt.x>=src.cols||pt.y<0||pt.y>=src.rows)
		return false;
	else
		return true;
}

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

void EyelidFeatureDetection(const Mat &Src , vector<Point> &upperEyelid_feature , vector<Point> &lowerEyelid_feature 
	, const int &eyeRegionCenter_y , const Mat &Sclera_mask)
{		
	//------------------------Upper Eyelid----------------------------//				
	//Upper Eyelid	
	Point startPoint;
	Point leftStartPoint(0 , eyeRegionCenter_y);
	Point rightStartPoint(Src.cols - 1 , eyeRegionCenter_y);	
	

	for(int j=leftStartPoint.x ; j<rightStartPoint.x; j+=Eupper_sourceFeature_inter_range_phase2){
		startPoint = Point(j , eyeRegionCenter_y);
		
		for(int k=0;k<Eupper_numSweep;++k){
			int angle_start_upper_phase2 = -90 - Eupper_angle_range_phase2;
			int angle_end_upper_phase2 = -90 + Eupper_angle_range_phase2;;
			int num_line_upper_phase2 = (angle_end_upper_phase2 - angle_start_upper_phase2)/(float)Eupper_inter_angle_phase2 +1;
			bool *check_line_halting_upper_phase2 = new bool [num_line_upper_phase2]();
			//int *count_NumOfFeature_line_upper_phase2 = new int [num_line_upper_phase2]();	
			//int *count_NumOfFeatureIsFtPtsPass_phase2 = new int [num_line_upper_phase2]();	
			bool finish_upper_phase2 = false;
			float radius_upper_phase2 = Eradius_initial;

			while(!finish_upper_phase2){	
				finish_upper_phase2 = true;
				for(int i=angle_start_upper_phase2 , count_linehalt_pos = 0;i<angle_end_upper_phase2+1; i+=Eupper_inter_angle_phase2 , ++count_linehalt_pos){
					if(check_line_halting_upper_phase2[count_linehalt_pos]==true)	continue;

					float pre_radius = radius_upper_phase2-EdistanceofFeature;
					int cal_x = cos(i*angle_multiply)*radius_upper_phase2+startPoint.x;
					int cal_y = sin(i*angle_multiply)*radius_upper_phase2+startPoint.y;
					int cal_pre_x = cos(i*angle_multiply)*(pre_radius)+startPoint.x;
					int cal_pre_y = sin(i*angle_multiply)*(pre_radius)+startPoint.y;
					Point point_get;
					point_get.x = (cal_x+cal_pre_x)/2;
					point_get.y = (cal_y+cal_pre_y)/2;
					
					if(checkpoint(point_get , Sclera_mask)){
						if(Sclera_mask.at<uchar>(point_get.y , point_get.x)==255)continue;
					}else{
						check_line_halting_upper_phase2[count_linehalt_pos]=true;
						continue;
					}

					if(checkpoint(Point(cal_x , cal_y) , Src) && checkpoint(Point(cal_pre_x , cal_pre_y) , Src)){//in image scale && not being used					
						if(fabs((float)Src.at<uchar>(cal_y , cal_x)-(float)Src.at<uchar>(cal_pre_y , cal_pre_x))<Eupper_gradient_threshold_phase2){
							continue;
						}else{					
							//if(count_NumOfFeatureIsFtPtsPass_phase2[count_linehalt_pos]>=Eupper_number_PassFeaturePts_perline_phase2){
							//	upperEyelid_feature.push_back(point_get);	

							//	
							//	++count_NumOfFeature_line_upper_phase2[count_linehalt_pos];
							//	if(count_NumOfFeature_line_upper_phase2[count_linehalt_pos]==Eupper_number_feature_perline_phase2)
							//		check_line_halting_upper_phase2[count_linehalt_pos]=true;
							//}// Pass Eupper_number_PassFeaturePts_perline_phase2 Feature Points , then count one Feature Points		

							//++count_NumOfFeatureIsFtPtsPass_phase2[count_linehalt_pos];

							//	upperEyelid_feature.push_back(point_get);	

							upperEyelid_feature.push_back(point_get);
							check_line_halting_upper_phase2[count_linehalt_pos] = true;				
						}
					}else if(checkpoint(Point(cal_x , cal_y) , Src) && !(checkpoint(Point(cal_pre_x , cal_pre_y) , Src))){
						continue;
					}else{//not in image scale
						check_line_halting_upper_phase2[count_linehalt_pos]=true;
					}			
				}
				radius_upper_phase2+=Eradius_initial;		
				for(int i=0;i<num_line_upper_phase2;++i){
					if(check_line_halting_upper_phase2[i]==false){
						finish_upper_phase2 = false;
						break;
					}
				}
			}//end while			
		}//end for		
	}//end for		

	//------------------------Lower Eyelid----------------------------//					
	//Lower Eyelid
	for(int j=leftStartPoint.x ; j<rightStartPoint.x; j+=Elower_sourceFeature_inter_range_phase2){
		startPoint = Point(j , eyeRegionCenter_y);

		for(int k=0;k<Elower_numSweep;++k){			
			int angle_start_lower_phase2 = 90 - Elower_angle_range_phase2;
			int angle_end_lower_phase2 = 90 + Elower_angle_range_phase2;;
			int num_line_lower_phase2 = (angle_end_lower_phase2 - angle_start_lower_phase2)/(float)Elower_inter_angle_phase2 +1;
			bool *check_line_halting_lower_phase2 = new bool [num_line_lower_phase2]();
			//int *count_NumOfFeature_line_lower_phase2 = new int [num_line_lower_phase2]();	
			//int *count_NumOfFeatureIsFtPtsPass_phase2 = new int [num_line_lower_phase2]();	
			bool finish_lower_phase2 = false;
			float radius_lower_phase2 = Eradius_initial;

			while(!finish_lower_phase2){	
				finish_lower_phase2 = true;
				for(int i=angle_start_lower_phase2 , count_linehalt_pos = 0;i<angle_end_lower_phase2+1; i+=Elower_inter_angle_phase2 , ++count_linehalt_pos){
					if(check_line_halting_lower_phase2[count_linehalt_pos]==true)	continue;

					float pre_radius = radius_lower_phase2-EdistanceofFeature;
					int cal_x = cos(i*angle_multiply)*radius_lower_phase2+startPoint.x;
					int cal_y = sin(i*angle_multiply)*radius_lower_phase2+startPoint.y;
					int cal_pre_x = cos(i*angle_multiply)*(pre_radius)+startPoint.x;
					int cal_pre_y = sin(i*angle_multiply)*(pre_radius)+startPoint.y;
					Point point_get;
					point_get.x = (cal_x+cal_pre_x)/2;
					point_get.y = (cal_y+cal_pre_y)/2;
					
					if(checkpoint(point_get , Sclera_mask)){
						if(Sclera_mask.at<uchar>(point_get.y , point_get.x)==255)continue;
					}else{
						check_line_halting_lower_phase2[count_linehalt_pos]=true;
						continue;
					}

					if(checkpoint(Point(cal_x , cal_y) , Src) && checkpoint(Point(cal_pre_x , cal_pre_y) , Src)){//in image scale && not being used					
						if(fabs((float)Src.at<uchar>(cal_y , cal_x)-(float)Src.at<uchar>(cal_pre_y , cal_pre_x))<Elower_gradient_threshold_phase2){
							continue;
						}else{					
							//if(count_NumOfFeatureIsFtPtsPass_phase2[count_linehalt_pos]>=Elower_number_PassFeaturePts_perline_phase2){
							//	lowerEyelid_feature.push_back(point_get);		
							//
							//	++count_NumOfFeature_line_lower_phase2[count_linehalt_pos];
							//	if(count_NumOfFeature_line_lower_phase2[count_linehalt_pos]==Elower_number_feature_perline_phase2)
							//		check_line_halting_lower_phase2[count_linehalt_pos]=true;
							//}// Pass Elower_number_PassFeaturePts_perline_phase2 Feature Points , then count one Feature Points		

							//++count_NumOfFeatureIsFtPtsPass_phase2[count_linehalt_pos];
						
							lowerEyelid_feature.push_back(point_get);
							check_line_halting_lower_phase2[count_linehalt_pos] = true;

						}
					}else if(checkpoint(Point(cal_x , cal_y) , Src) && !(checkpoint(Point(cal_pre_x , cal_pre_y) , Src))){
						continue;
					}else{//not in image scale
						check_line_halting_lower_phase2[count_linehalt_pos]=true;
					}			
				}
				radius_lower_phase2+=Eradius_initial;		
				for(int i=0;i<num_line_lower_phase2;++i){
					if(check_line_halting_lower_phase2[i]==false){
						finish_lower_phase2 = false;
						break;
					}
				}
			}//end while			
		}//end for		
	}//end for
}

class Parallel_EyelidFeatureDetection : public cv::ParallelLoopBody
{

private:
	const Mat &Src;
	const Mat &Sclera_mask;
	vector<Point> *upperEyelid_feture_arr;
	vector<Point> *lowerEyelid_feture_arr;
	int diff;
	int *start_loc;
	int *end_loc;
	const int eyeRegionCenter_y;
public:
	Parallel_EyelidFeatureDetection(const Mat &Src, vector<Point> *upperEyelid_feture_arr, vector<Point> *lowerEyelid_feture_arr,
		int *start_loc, int *end_loc, const int &eyeRegionCenter_y, const Mat &Sclera_mask, int diffVal)
		: Src(Src), Sclera_mask(Sclera_mask), upperEyelid_feture_arr(upperEyelid_feture_arr), lowerEyelid_feture_arr(lowerEyelid_feture_arr)
		, start_loc(start_loc), end_loc(end_loc), eyeRegionCenter_y(eyeRegionCenter_y), diff(diffVal){}

	virtual void operator()(const cv::Range& range) const
	{

		//#pragma omp parallel for		
		for (int i = range.start; i < range.end; ++i)
		{
			/* divide image in 'diff' number
			of parts and process simultaneously */
			//------------------------Upper Eyelid----------------------------//				
			//Upper Eyelid	
			for (int j = start_loc[i]; j <= end_loc[i]; j += Eupper_sourceFeature_inter_range_phase2) {
				Point startPoint = Point(j, eyeRegionCenter_y);

				for (int k = 0; k<Eupper_numSweep; ++k) {
					int angle_start_upper_phase2 = -90 - Eupper_angle_range_phase2;
					int angle_end_upper_phase2 = -90 + Eupper_angle_range_phase2;;
					int num_line_upper_phase2 = (angle_end_upper_phase2 - angle_start_upper_phase2) / (float)Eupper_inter_angle_phase2 + 1;
					bool *check_line_halting_upper_phase2 = new bool[num_line_upper_phase2]();
					//int *count_NumOfFeature_line_upper_phase2 = new int [num_line_upper_phase2]();	
					//int *count_NumOfFeatureIsFtPtsPass_phase2 = new int [num_line_upper_phase2]();	
					bool finish_upper_phase2 = false;
					float radius_upper_phase2 = Eradius_initial;

					while (!finish_upper_phase2) {
						finish_upper_phase2 = true;
						for (int k = angle_start_upper_phase2, count_linehalt_pos = 0; k<angle_end_upper_phase2 + 1; k += Eupper_inter_angle_phase2, ++count_linehalt_pos) {
							if (check_line_halting_upper_phase2[count_linehalt_pos] == true)	continue;

							float pre_radius = radius_upper_phase2 - EdistanceofFeature;
							int cal_x = cos(k*angle_multiply)*radius_upper_phase2 + startPoint.x;
							int cal_y = sin(k*angle_multiply)*radius_upper_phase2 + startPoint.y;
							int cal_pre_x = cos(k*angle_multiply)*(pre_radius)+startPoint.x;
							int cal_pre_y = sin(k*angle_multiply)*(pre_radius)+startPoint.y;
							Point point_get;
							point_get.x = (cal_x + cal_pre_x) / 2;
							point_get.y = (cal_y + cal_pre_y) / 2;

							if (checkpoint(point_get, Sclera_mask)) {
								if (Sclera_mask.at<uchar>(point_get.y, point_get.x) == 255)continue;
							}else{
								check_line_halting_upper_phase2[count_linehalt_pos] = true;
								continue;
							}

							if (checkpoint(Point(cal_x, cal_y), Src) && checkpoint(Point(cal_pre_x, cal_pre_y), Src)) {//in image scale && not being used					
								if (fabs((float)Src.at<uchar>(cal_y, cal_x) - (float)Src.at<uchar>(cal_pre_y, cal_pre_x))<Eupper_gradient_threshold_phase2) {
									continue;
								}else {
									//if(count_NumOfFeatureIsFtPtsPass_phase2[count_linehalt_pos]>=Eupper_number_PassFeaturePts_perline_phase2){
									//	upperEyelid_feature.push_back(point_get);	

									//	
									//	++count_NumOfFeature_line_upper_phase2[count_linehalt_pos];
									//	if(count_NumOfFeature_line_upper_phase2[count_linehalt_pos]==Eupper_number_feature_perline_phase2)
									//		check_line_halting_upper_phase2[count_linehalt_pos]=true;
									//}// Pass Eupper_number_PassFeaturePts_perline_phase2 Feature Points , then count one Feature Points		

									//++count_NumOfFeatureIsFtPtsPass_phase2[count_linehalt_pos];

									//	upperEyelid_feature.push_back(point_get);	

									upperEyelid_feture_arr[i].push_back(point_get);
									check_line_halting_upper_phase2[count_linehalt_pos] = true;
								}
							}else if (checkpoint(Point(cal_x, cal_y), Src) && !(checkpoint(Point(cal_pre_x, cal_pre_y), Src))) {
								continue;
							}else {//not in image scale
								check_line_halting_upper_phase2[count_linehalt_pos] = true;
							}
						}
						radius_upper_phase2 += Eradius_initial;
						for (int k = 0; k<num_line_upper_phase2; ++k) {
							if (check_line_halting_upper_phase2[k] == false) {
								finish_upper_phase2 = false;
								break;
							}
						}
					}//end while			
				}//end for		
			}//end for		

			 //------------------------Lower Eyelid----------------------------//					
			 //Lower Eyelid
			for (int j = start_loc[i]; j <= end_loc[i]; j += Elower_sourceFeature_inter_range_phase2) {
				Point startPoint = Point(j, eyeRegionCenter_y);

				for (int k = 0; k<Elower_numSweep; ++k) {
					int angle_start_lower_phase2 = 90 - Elower_angle_range_phase2;
					int angle_end_lower_phase2 = 90 + Elower_angle_range_phase2;;
					int num_line_lower_phase2 = (angle_end_lower_phase2 - angle_start_lower_phase2) / (float)Elower_inter_angle_phase2 + 1;
					bool *check_line_halting_lower_phase2 = new bool[num_line_lower_phase2]();
					//int *count_NumOfFeature_line_lower_phase2 = new int [num_line_lower_phase2]();	
					//int *count_NumOfFeatureIsFtPtsPass_phase2 = new int [num_line_lower_phase2]();	
					bool finish_lower_phase2 = false;
					float radius_lower_phase2 = Eradius_initial;

					while (!finish_lower_phase2) {
						finish_lower_phase2 = true;
						for (int k = angle_start_lower_phase2, count_linehalt_pos = 0; k<angle_end_lower_phase2 + 1; k += Elower_inter_angle_phase2, ++count_linehalt_pos) {
							if (check_line_halting_lower_phase2[count_linehalt_pos] == true)	continue;

							float pre_radius = radius_lower_phase2 - EdistanceofFeature;
							int cal_x = cos(k*angle_multiply)*radius_lower_phase2 + startPoint.x;
							int cal_y = sin(k*angle_multiply)*radius_lower_phase2 + startPoint.y;
							int cal_pre_x = cos(k*angle_multiply)*(pre_radius)+startPoint.x;
							int cal_pre_y = sin(k*angle_multiply)*(pre_radius)+startPoint.y;
							Point point_get;
							point_get.x = (cal_x + cal_pre_x) / 2;
							point_get.y = (cal_y + cal_pre_y) / 2;

							if (checkpoint(point_get, Sclera_mask)) {
								if (Sclera_mask.at<uchar>(point_get.y, point_get.x) == 255)continue;
							}
							else {
								check_line_halting_lower_phase2[count_linehalt_pos] = true;
								continue;
							}

							if (checkpoint(Point(cal_x, cal_y), Src) && checkpoint(Point(cal_pre_x, cal_pre_y), Src)) {//in image scale && not being used					
								if (fabs((float)Src.at<uchar>(cal_y, cal_x) - (float)Src.at<uchar>(cal_pre_y, cal_pre_x))<Elower_gradient_threshold_phase2) {
									continue;
								}
								else {
									//if(count_NumOfFeatureIsFtPtsPass_phase2[count_linehalt_pos]>=Elower_number_PassFeaturePts_perline_phase2){
									//	lowerEyelid_feature.push_back(point_get);		
									//
									//	++count_NumOfFeature_line_lower_phase2[count_linehalt_pos];
									//	if(count_NumOfFeature_line_lower_phase2[count_linehalt_pos]==Elower_number_feature_perline_phase2)
									//		check_line_halting_lower_phase2[count_linehalt_pos]=true;
									//}// Pass Elower_number_PassFeaturePts_perline_phase2 Feature Points , then count one Feature Points		

									//++count_NumOfFeatureIsFtPtsPass_phase2[count_linehalt_pos];

									lowerEyelid_feture_arr[i].push_back(point_get);
									check_line_halting_lower_phase2[count_linehalt_pos] = true;

								}
							}
							else if (checkpoint(Point(cal_x, cal_y), Src) && !(checkpoint(Point(cal_pre_x, cal_pre_y), Src))) {
								continue;
							}
							else {//not in image scale
								check_line_halting_lower_phase2[count_linehalt_pos] = true;
							}
						}
						radius_lower_phase2 += Eradius_initial;
						for (int k = 0; k<num_line_lower_phase2; ++k) {
							if (check_line_halting_lower_phase2[k] == false) {
								finish_lower_phase2 = false;
								break;
							}
						}
					}//end while			
				}//end for		
			}//end for
		}//end for i=range.start
	}
};

class Parallel_EyelidFeatureDetectionLock : public cv::ParallelLoopBody
{

private:
	const Mat &Src;
	const Mat &Sclera_mask;
	vector<Point> &upperEyelid_feture_arr;
	vector<Point> &lowerEyelid_feture_arr;
	int diff;
	int *start_loc;
	int *end_loc;
	const int eyeRegionCenter_y;
	std::mutex &mtx;
public:
	Parallel_EyelidFeatureDetectionLock(const Mat &Src, vector<Point> &upperEyelid_feture_arr, vector<Point> &lowerEyelid_feture_arr,
		int *start_loc, int *end_loc, const int &eyeRegionCenter_y, const Mat &Sclera_mask, std::mutex &mtx, int diffVal)
		: Src(Src), Sclera_mask(Sclera_mask), upperEyelid_feture_arr(upperEyelid_feture_arr), lowerEyelid_feture_arr(lowerEyelid_feture_arr)
		, start_loc(start_loc), end_loc(end_loc), eyeRegionCenter_y(eyeRegionCenter_y), mtx(mtx), diff(diffVal) {}

	virtual void operator()(const cv::Range& range) const
	{

		//#pragma omp parallel for		
		for (int i = range.start; i < range.end; ++i)
		{
			/* divide image in 'diff' number
			of parts and process simultaneously */
			//------------------------Upper Eyelid----------------------------//				
			//Upper Eyelid	
			for (int j = start_loc[i]; j <= end_loc[i]; j += Eupper_sourceFeature_inter_range_phase2) {
				Point startPoint = Point(j, eyeRegionCenter_y);

				for (int k = 0; k<Eupper_numSweep; ++k) {
					int angle_start_upper_phase2 = -90 - Eupper_angle_range_phase2;
					int angle_end_upper_phase2 = -90 + Eupper_angle_range_phase2;;
					int num_line_upper_phase2 = (angle_end_upper_phase2 - angle_start_upper_phase2) / (float)Eupper_inter_angle_phase2 + 1;
					bool *check_line_halting_upper_phase2 = new bool[num_line_upper_phase2]();
					//int *count_NumOfFeature_line_upper_phase2 = new int [num_line_upper_phase2]();	
					//int *count_NumOfFeatureIsFtPtsPass_phase2 = new int [num_line_upper_phase2]();	
					bool finish_upper_phase2 = false;
					float radius_upper_phase2 = Eradius_initial;

					while (!finish_upper_phase2) {
						finish_upper_phase2 = true;
						for (int k = angle_start_upper_phase2, count_linehalt_pos = 0; k<angle_end_upper_phase2 + 1; k += Eupper_inter_angle_phase2, ++count_linehalt_pos) {
							if (check_line_halting_upper_phase2[count_linehalt_pos] == true)	continue;

							float pre_radius = radius_upper_phase2 - EdistanceofFeature;
							int cal_x = cos(k*angle_multiply)*radius_upper_phase2 + startPoint.x;
							int cal_y = sin(k*angle_multiply)*radius_upper_phase2 + startPoint.y;
							int cal_pre_x = cos(k*angle_multiply)*(pre_radius)+startPoint.x;
							int cal_pre_y = sin(k*angle_multiply)*(pre_radius)+startPoint.y;
							Point point_get;
							point_get.x = (cal_x + cal_pre_x) / 2;
							point_get.y = (cal_y + cal_pre_y) / 2;

							if (checkpoint(point_get, Sclera_mask)) {
								if (Sclera_mask.at<uchar>(point_get.y, point_get.x) == 255)continue;
							}
							else {
								check_line_halting_upper_phase2[count_linehalt_pos] = true;
								continue;
							}

							if (checkpoint(Point(cal_x, cal_y), Src) && checkpoint(Point(cal_pre_x, cal_pre_y), Src)) {//in image scale && not being used					
								if (fabs((float)Src.at<uchar>(cal_y, cal_x) - (float)Src.at<uchar>(cal_pre_y, cal_pre_x))<Eupper_gradient_threshold_phase2) {
									continue;
								}
								else {
									//if(count_NumOfFeatureIsFtPtsPass_phase2[count_linehalt_pos]>=Eupper_number_PassFeaturePts_perline_phase2){
									//	upperEyelid_feature.push_back(point_get);	

									//	
									//	++count_NumOfFeature_line_upper_phase2[count_linehalt_pos];
									//	if(count_NumOfFeature_line_upper_phase2[count_linehalt_pos]==Eupper_number_feature_perline_phase2)
									//		check_line_halting_upper_phase2[count_linehalt_pos]=true;
									//}// Pass Eupper_number_PassFeaturePts_perline_phase2 Feature Points , then count one Feature Points		

									//++count_NumOfFeatureIsFtPtsPass_phase2[count_linehalt_pos];

									//	upperEyelid_feature.push_back(point_get);	

									mtx.lock();
									upperEyelid_feture_arr.push_back(point_get);
									mtx.unlock();
									check_line_halting_upper_phase2[count_linehalt_pos] = true;
								}
							}
							else if (checkpoint(Point(cal_x, cal_y), Src) && !(checkpoint(Point(cal_pre_x, cal_pre_y), Src))) {
								continue;
							}
							else {//not in image scale
								check_line_halting_upper_phase2[count_linehalt_pos] = true;
							}
						}
						radius_upper_phase2 += Eradius_initial;
						for (int k = 0; k<num_line_upper_phase2; ++k) {
							if (check_line_halting_upper_phase2[k] == false) {
								finish_upper_phase2 = false;
								break;
							}
						}
					}//end while			
				}//end for		
			}//end for		

			 //------------------------Lower Eyelid----------------------------//					
			 //Lower Eyelid
			for (int j = start_loc[i]; j <= end_loc[i]; j += Elower_sourceFeature_inter_range_phase2) {
				Point startPoint = Point(j, eyeRegionCenter_y);

				for (int k = 0; k<Elower_numSweep; ++k) {
					int angle_start_lower_phase2 = 90 - Elower_angle_range_phase2;
					int angle_end_lower_phase2 = 90 + Elower_angle_range_phase2;;
					int num_line_lower_phase2 = (angle_end_lower_phase2 - angle_start_lower_phase2) / (float)Elower_inter_angle_phase2 + 1;
					bool *check_line_halting_lower_phase2 = new bool[num_line_lower_phase2]();
					//int *count_NumOfFeature_line_lower_phase2 = new int [num_line_lower_phase2]();	
					//int *count_NumOfFeatureIsFtPtsPass_phase2 = new int [num_line_lower_phase2]();	
					bool finish_lower_phase2 = false;
					float radius_lower_phase2 = Eradius_initial;

					while (!finish_lower_phase2) {
						finish_lower_phase2 = true;
						for (int k = angle_start_lower_phase2, count_linehalt_pos = 0; k<angle_end_lower_phase2 + 1; k += Elower_inter_angle_phase2, ++count_linehalt_pos) {
							if (check_line_halting_lower_phase2[count_linehalt_pos] == true)	continue;

							float pre_radius = radius_lower_phase2 - EdistanceofFeature;
							int cal_x = cos(k*angle_multiply)*radius_lower_phase2 + startPoint.x;
							int cal_y = sin(k*angle_multiply)*radius_lower_phase2 + startPoint.y;
							int cal_pre_x = cos(k*angle_multiply)*(pre_radius)+startPoint.x;
							int cal_pre_y = sin(k*angle_multiply)*(pre_radius)+startPoint.y;
							Point point_get;
							point_get.x = (cal_x + cal_pre_x) / 2;
							point_get.y = (cal_y + cal_pre_y) / 2;

							if (checkpoint(point_get, Sclera_mask)) {
								if (Sclera_mask.at<uchar>(point_get.y, point_get.x) == 255)continue;
							}
							else {
								check_line_halting_lower_phase2[count_linehalt_pos] = true;
								continue;
							}

							if (checkpoint(Point(cal_x, cal_y), Src) && checkpoint(Point(cal_pre_x, cal_pre_y), Src)) {//in image scale && not being used					
								if (fabs((float)Src.at<uchar>(cal_y, cal_x) - (float)Src.at<uchar>(cal_pre_y, cal_pre_x))<Elower_gradient_threshold_phase2) {
									continue;
								}
								else {
									//if(count_NumOfFeatureIsFtPtsPass_phase2[count_linehalt_pos]>=Elower_number_PassFeaturePts_perline_phase2){
									//	lowerEyelid_feature.push_back(point_get);		
									//
									//	++count_NumOfFeature_line_lower_phase2[count_linehalt_pos];
									//	if(count_NumOfFeature_line_lower_phase2[count_linehalt_pos]==Elower_number_feature_perline_phase2)
									//		check_line_halting_lower_phase2[count_linehalt_pos]=true;
									//}// Pass Elower_number_PassFeaturePts_perline_phase2 Feature Points , then count one Feature Points		

									//++count_NumOfFeatureIsFtPtsPass_phase2[count_linehalt_pos];

									mtx.lock();
									lowerEyelid_feture_arr.push_back(point_get);
									mtx.unlock();
									check_line_halting_lower_phase2[count_linehalt_pos] = true;

								}
							}
							else if (checkpoint(Point(cal_x, cal_y), Src) && !(checkpoint(Point(cal_pre_x, cal_pre_y), Src))) {
								continue;
							}
							else {//not in image scale
								check_line_halting_lower_phase2[count_linehalt_pos] = true;
							}
						}
						radius_lower_phase2 += Eradius_initial;
						for (int k = 0; k<num_line_lower_phase2; ++k) {
							if (check_line_halting_lower_phase2[k] == false) {
								finish_lower_phase2 = false;
								break;
							}
						}
					}//end while			
				}//end for		
			}//end for
		}//end for i=range.start
	}
};

void ParalleEyelidFeatureDetection(const Mat &Src, vector<Point> &upperEyelid_feature, vector<Point> &lowerEyelid_feature
	, const int &eyeRegionCenter_y, const Mat &Sclera_mask, const int &thread_num
	, vector<double> &time_eye_position_detection_eyelid_feature_detection_district5_core
	, vector<double> &time_eye_position_detection_eyelid_feature_detection_district5_merge) {
	int *start_loc = new int[thread_num]();
	int *end_loc = new int[thread_num]();
	int avg_work_load = Src.cols / thread_num;
	int extra_work_load = Src.cols % thread_num;
	vector<Point> *upperEyelid_feture_arr = new vector<Point>[thread_num];
	vector<Point> *lowerEyelid_feture_arr = new vector<Point>[thread_num];

	for (int i = 0; i<thread_num; ++i) {
		int my_rank_load = (i<(extra_work_load)) ? (avg_work_load + 1) : avg_work_load;
		start_loc[i] = (i == 0) ? 0 : end_loc[i - 1] + 1;
		end_loc[i] = start_loc[i] + (my_rank_load - 1);
	}
	double time_1 = getTickCount();
	cv::parallel_for_(cv::Range(0, thread_num), Parallel_EyelidFeatureDetection(Src, upperEyelid_feture_arr, lowerEyelid_feture_arr,
		start_loc, end_loc, eyeRegionCenter_y, Sclera_mask, thread_num));

	double time_2 = getTickCount();
	for (int i = 0; i < thread_num; ++i) {
		upperEyelid_feature.insert(std::end(upperEyelid_feature), std::begin(upperEyelid_feture_arr[i]), std::end(upperEyelid_feture_arr[i]));
		lowerEyelid_feature.insert(std::end(lowerEyelid_feature), std::begin(lowerEyelid_feture_arr[i]), std::end(lowerEyelid_feture_arr[i]));
	}
	double time_3 = getTickCount();

	time_eye_position_detection_eyelid_feature_detection_district5_core.push_back(time_2 - time_1);
	time_eye_position_detection_eyelid_feature_detection_district5_merge.push_back(time_3 - time_2);

	delete [] start_loc;
	delete [] end_loc;
	delete [] upperEyelid_feture_arr;
	delete [] lowerEyelid_feture_arr;
}

void ParalleEyelidFeatureDetectionLock(const Mat &Src, vector<Point> &upperEyelid_feature, vector<Point> &lowerEyelid_feature
	, const int &eyeRegionCenter_y, const Mat &Sclera_mask, const int &thread_num
	, vector<double> &time_eye_position_detection_eyelid_feature_detection_district5_core
	, vector<double> &time_eye_position_detection_eyelid_feature_detection_district5_merge) {
	int *start_loc = new int[thread_num]();
	int *end_loc = new int[thread_num]();
	int avg_work_load = Src.cols / thread_num;
	int extra_work_load = Src.cols % thread_num;
	std::mutex mtx;

	for (int i = 0; i<thread_num; ++i) {
		int my_rank_load = (i<(extra_work_load)) ? (avg_work_load + 1) : avg_work_load;
		start_loc[i] = (i == 0) ? 0 : end_loc[i - 1] + 1;
		end_loc[i] = start_loc[i] + (my_rank_load - 1);
	}
	double time_1 = getTickCount();
	cv::parallel_for_(cv::Range(0, thread_num), Parallel_EyelidFeatureDetectionLock(Src, upperEyelid_feature, lowerEyelid_feature,
		start_loc, end_loc, eyeRegionCenter_y, Sclera_mask, mtx, thread_num));

	double time_2 = getTickCount();
	//for (int i = 0; i < thread_num; ++i) {
	//	upperEyelid_feature.insert(std::end(upperEyelid_feature), std::begin(upperEyelid_feture_arr[i]), std::end(upperEyelid_feture_arr[i]));
	//	lowerEyelid_feature.insert(std::end(lowerEyelid_feature), std::begin(lowerEyelid_feture_arr[i]), std::end(lowerEyelid_feture_arr[i]));
	//}
	double time_3 = getTickCount();

	time_eye_position_detection_eyelid_feature_detection_district5_core.push_back(time_2 - time_1);
	time_eye_position_detection_eyelid_feature_detection_district5_merge.push_back(time_3 - time_2);

	delete[] start_loc;
	delete[] end_loc;
}
