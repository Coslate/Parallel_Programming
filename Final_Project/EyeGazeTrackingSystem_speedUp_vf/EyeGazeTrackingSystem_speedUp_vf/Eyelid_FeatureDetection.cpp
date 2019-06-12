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
	, const int &frame_number , const int &eyeRegionCenter_y , const Mat &Sclera_mask)
{		
	//------------------------Upper Eyelid----------------------------//			
	Mat Check_Already_IsFeature = Mat::zeros(Src.rows , Src.cols , CV_8UC1);	

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
			int *count_NumOfFeature_line_upper_phase2 = new int [num_line_upper_phase2]();	
			int *count_NumOfFeatureIsFtPtsPass_phase2 = new int [num_line_upper_phase2]();	
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
							if(count_NumOfFeatureIsFtPtsPass_phase2[count_linehalt_pos]>=Eupper_number_PassFeaturePts_perline_phase2){
								upperEyelid_feature.push_back(point_get);	

								
								++count_NumOfFeature_line_upper_phase2[count_linehalt_pos];
								if(count_NumOfFeature_line_upper_phase2[count_linehalt_pos]==Eupper_number_feature_perline_phase2)
									check_line_halting_upper_phase2[count_linehalt_pos]=true;
							}// Pass Eupper_number_PassFeaturePts_perline_phase2 Feature Points , then count one Feature Points		

							++count_NumOfFeatureIsFtPtsPass_phase2[count_linehalt_pos];
						
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
			int *count_NumOfFeature_line_lower_phase2 = new int [num_line_lower_phase2]();	
			int *count_NumOfFeatureIsFtPtsPass_phase2 = new int [num_line_lower_phase2]();	
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
							if(count_NumOfFeatureIsFtPtsPass_phase2[count_linehalt_pos]>=Elower_number_PassFeaturePts_perline_phase2){
								lowerEyelid_feature.push_back(point_get);		
							
								++count_NumOfFeature_line_lower_phase2[count_linehalt_pos];
								if(count_NumOfFeature_line_lower_phase2[count_linehalt_pos]==Elower_number_feature_perline_phase2)
									check_line_halting_lower_phase2[count_linehalt_pos]=true;
							}// Pass Elower_number_PassFeaturePts_perline_phase2 Feature Points , then count one Feature Points		

							++count_NumOfFeatureIsFtPtsPass_phase2[count_linehalt_pos];
						
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
