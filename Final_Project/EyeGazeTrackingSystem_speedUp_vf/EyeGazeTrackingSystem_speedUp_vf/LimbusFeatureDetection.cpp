#include "Limbus_FeatureDetection.h"
#include<vector>
#include<cmath>
#include <omp.h>
#include <fstream>


inline bool checkpoint(const int x ,const int y){
	if(x<0||x>=FRAMEW||y<0||y>=FRAMEH)
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

inline const float& DistanceCaculateEuclidean(const Point2f &x1 , const Point2f &x2){
	Point2f vectorLineX1X2(x2.x - x1.x , x2.y - x1.y);
	return sqrtf(vectorLineX1X2.x*vectorLineX1X2.x + vectorLineX1X2.y*vectorLineX1X2.y);
}

inline double DistanceOfPoints(const Point x , const Point y){
	return(powf(powf(x.x - y.x , 2.f)+pow(x.y - y.y , 2.f) , 0.5));
}
inline float DistanceCaculate(const Point &x1 , const Point &x2){
	return max(abs((x1.x - x2.x)) , abs((x1.y - x2.y)))+0.5*min(abs((x1.x - x2.x)) , abs((x1.y - x2.y)));
}

inline bool ISinEyelidRegion(const Point &in_p , const Mat &eyelid_removal){
	if((int)eyelid_removal.at<uchar>(in_p.y , in_p.x)==255){
		return true;
	}else{
		return false;
	}
}

inline void feature_distance_filter(const vector<Point> &feature_pre_filter , vector<Point> &feature , const double c_x
	 , const double c_y){

	const int feature_num = feature_pre_filter.size();
	if(feature_num==0){
		feature = feature_pre_filter;
	}else{
		double dist_var;
		double sum_var = 0;
		double dist_mean = 0;
		double *dist_f = new double [feature_num]();
		
		for(int i=0;i<feature_num;++i){
			dist_f[i] = sqrtf(powf(feature_pre_filter[i].x-c_x , 2.f)+powf(feature_pre_filter[i].y-c_y , 2.f));
		}
		for(int i=0;i<feature_num;++i){
			dist_mean+=dist_f[i];
		}
		
		dist_mean/=feature_num;

		for(int i=0;i<feature_num;++i){
			sum_var+=powf(dist_f[i]-dist_mean , 2.f);
		}
		sum_var/=feature_num;
		dist_var = sqrtf(sum_var);

		for(int i=0;i<feature_num;++i){
			if(dist_f[i]>dist_mean+1.5*dist_var || dist_f[i]<dist_mean-1.5*dist_var){
				continue;
			}else{
				feature.push_back(feature_pre_filter.at(i));
			}		
		}	
	}

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

  line(image,pt1,pt2,color,4,8);
  line(image,pt3,pt4,color,4,8);
}
inline void CheckError3(const int frame_num , const Mat &in , const vector<Point> &pt 
	, const vector<Point> &pt2){
	Mat test = in.clone();
	if(frame_num==28){
		for(int i=0;i<pt.size();++i){
			Draw_Cross(test, pt[i].x, pt[i].y, 20, 20, Scalar(100 , 100 , 255));					
		}
		for(int i=0;i<pt2.size();++i){
			Draw_Cross(test, pt2[i].x, pt2[i].y, 20, 20, Scalar(255 , 255 , 255));					
		}
		
	}
	imshow("test2" , test);
}

inline void CheckError2(const int frame_num , const Mat &in , const vector<Point> &pt){
	Mat test = in.clone();
	if(frame_num==28){
		for(int i=0;i<pt.size();++i)
			Draw_Cross(test, pt[i].x, pt[i].y, 20, 20, Scalar(100 , 100 , 255));		
	}
	imshow("test2" , test);
}
inline void CheckError1(const int frame_num , const Mat &in , const vector<Point> &start_pt){
	Mat test = in.clone();
	if(frame_num==28){
		for(int i=0;i<start_pt.size();++i)
			Draw_Cross(test, start_pt[i].x, start_pt[i].y, 20, 20, Scalar(255 , 100 , 255));//pt1_start		
	}
	imshow("test" , test);
}




void LimbusFeatureDetection(const Mat &in , vector<Point> &feature , const int frame_number , const int num_per_line 
	, Point & start_point , const bool setup_eye_start_point_man , const Mat &Iris_Mask, vector<double> &time_eye_position_detection_limbus_feature_detection_serial)
{		 		
	if(!setup_eye_start_point_man){
		start_point.x = in.cols/2;
		start_point.y = in.rows/2;
	}	
		

	
	float gradient_threshold = gradient_threshold_initial;	
	Point old_point;
	int count_cycle = 0;		
	float ang_mul = 180.f/PI;
	
	while(1){	
		feature.clear();
		vector<Point> feature_pre_filter;
		Mat CaculatedFeaturePoint = Mat::zeros(in.rows , in.cols , CV_8UC1);		

		bool Finish_stage1 = false;
		bool Finish_stage2 = false;	
		bool Finish_stage1_part1;
		bool Finish_stage1_part2;
		
		float radius = radius_stage1_initial;
		float center_x;
		float center_y;
		float sum_x = 0;
		float sum_y = 0;

		int num_line_part1 = (angle_stage1_end - angle_stage1_start)/(float)inter_angle_stage1_phase1 +1;
		int num_line_part2 = (angle_stage2_end - angle_stage2_start)/(float)inter_angle_stage1_phase2 +1;

		bool *check_line_halting_part1 = new bool[num_line_part1]();
		bool *check_line_halting_part2 = new bool[num_line_part2]();
		int *count_line_halting_part1 = new int[num_line_part1]();
		int *count_line_halting_part2 = new int[num_line_part2]();

		old_point = start_point;

		while(!Finish_stage1){//stage1 
			Finish_stage1 = true;
			Finish_stage1_part1 = true;
			Finish_stage1_part2 = true;
			
			for(int i=angle_stage1_start , angle_member = 0;i<angle_stage1_end+1; i += inter_angle_stage1_phase1 , ++angle_member){//phase 1
				//int angle_member = i-angle_stage1_start;
				if(check_line_halting_part1[angle_member]==false){
					int cal_x = cos(i*angle_multiply)*radius+start_point.x;
					int cal_y = sin(i*angle_multiply)*radius+start_point.y;
					int cal_pre_x = cos(i*angle_multiply)*(radius-distanceofFeature)+start_point.x;
					int cal_pre_y = sin(i*angle_multiply)*(radius-distanceofFeature)+start_point.y;
					Point point_get;
					point_get.x = (cal_x+cal_pre_x)/2;
					point_get.y = (cal_y+cal_pre_y)/2;

					if(checkpoint(point_get , Iris_Mask)){
						if(Iris_Mask.at<uchar>(point_get.y , point_get.x)==255)continue;
					}else{
						check_line_halting_part1[angle_member]=true;
						continue;
					}

					if(checkpoint(Point(cal_x , cal_y) , in) && checkpoint(Point(cal_pre_x , cal_pre_y) , in)){//in image scale	
						if(CaculatedFeaturePoint.at<uchar>(point_get.y , point_get.x)==1)continue;
						if(fabs((float)in.at<uchar>(cal_y , cal_x)-(float)in.at<uchar>(cal_pre_y , cal_pre_x))<gradient_threshold){
							continue;
						}else{
							++count_line_halting_part1[angle_member];

							feature_pre_filter.push_back(point_get);													
							CaculatedFeaturePoint.at<uchar>(point_get.y , point_get.x) = 1;						

							if(count_line_halting_part1[angle_member]>num_per_line-1)
								check_line_halting_part1[angle_member]=true;
						}
					}else{//not in image scale
						check_line_halting_part1[angle_member]=true;
					}
				}				
			}
			for(int i=angle_stage2_start , angle_member = 0;i<angle_stage2_end+1; i += inter_angle_stage1_phase2 , ++angle_member){// phase 2
				//int angle_member = i-angle_stage2_start;
				if(check_line_halting_part2[angle_member]==false){
					int cal_x = cos(i*angle_multiply)*radius+start_point.x;
					int cal_y = sin(i*angle_multiply)*radius+start_point.y;
					int cal_pre_x = cos(i*angle_multiply)*(radius-distanceofFeature)+start_point.x;
					int cal_pre_y = sin(i*angle_multiply)*(radius-distanceofFeature)+start_point.y;
					Point point_get;
					point_get.x = (cal_x+cal_pre_x)/2;
					point_get.y = (cal_y+cal_pre_y)/2;


					if(checkpoint(point_get , Iris_Mask)){
						if(Iris_Mask.at<uchar>(point_get.y , point_get.x)==255)continue;
					}else{
						check_line_halting_part2[angle_member]=true;
						continue;
					}

					if(checkpoint(Point(cal_x , cal_y) , in) && checkpoint(Point(cal_pre_x , cal_pre_y) , in)){//in image scale
						if(CaculatedFeaturePoint.at<uchar>(point_get.y , point_get.x)==1)continue;
						if(fabs((float)in.at<uchar>(cal_y , cal_x)-(float)in.at<uchar>(cal_pre_y , cal_pre_x))<gradient_threshold){
							continue;
						}else{							
		
							++count_line_halting_part2[angle_member];
							feature_pre_filter.push_back(point_get);																				
							CaculatedFeaturePoint.at<uchar>(point_get.y , point_get.x) = 1;

							if(count_line_halting_part2[angle_member]>num_per_line-1)
								check_line_halting_part2[angle_member]=true;
						}
					}else{//not in image scale
						check_line_halting_part2[angle_member]=true;
					}
				}		
			}
			radius+=radius_stage1_initial;
			for(int i=0;i<num_line_part1;++i){
				if(check_line_halting_part1[i]==false){
					Finish_stage1_part1 = false;
				}
			}

			for(int i=0;i<num_line_part2;++i){
				if(check_line_halting_part2[i]==false){
					Finish_stage1_part2 = false;
				}
			}

			if(Finish_stage1_part1 && Finish_stage1_part2){
				Finish_stage1 = true;
			}else{
				Finish_stage1 = false;
			}
		
			//cout<<"while stage 1 end"<<endl;
		}//end while stage1

		vector<Point> tmp;
		for(int i=0;i<feature_pre_filter.size();++i){
			tmp.push_back(feature_pre_filter[i]);
		}

		for(int i=0;i<tmp.size();++i){//stage2 , for every feature extracted from stage1 : 			
			int count_in = 0;//can be deleted
			int num = 2*angel_stage2_range/(float)inter_angle_stage2 + 1;
			bool *check_line_halting = new bool[num]();
			int *count_line_halting = new int [num]();			
			Point initial_point(tmp[i]);
			int angle_est = atan2((float)(start_point.y-initial_point.y),float(start_point.x-initial_point.x))*ang_mul;					
			radius = radius_stage2_initial;


			Finish_stage2 = false;				
			while(!Finish_stage2){//stage1 -30 degree ~ 30 degree 
				Finish_stage2 = true;										

				for(int angle_mem = (angle_est-angel_stage2_range) , angel_member = 0 ;angle_mem < (angle_est+angel_stage2_range)+1 ; angle_mem+=inter_angle_stage2 , ++angel_member){//-25~25degree
					if(check_line_halting[angel_member]==false){
						int cal_x = cos(angle_mem*angle_multiply)*radius+initial_point.x;
						int cal_y = sin(angle_mem*angle_multiply)*radius+initial_point.y;
						int cal_pre_x = cos(angle_mem*angle_multiply)*(radius-distanceofFeature)+initial_point.x;
						int cal_pre_y = sin(angle_mem*angle_multiply)*(radius-distanceofFeature)+initial_point.y;
						Point point_get;
						point_get.x = (cal_x+cal_pre_x)/2;
						point_get.y = (cal_y+cal_pre_y)/2;

						if(checkpoint(point_get , Iris_Mask)){
							if(Iris_Mask.at<uchar>(point_get.y , point_get.x)==255)continue;
						}else{
							check_line_halting[angel_member]=true;
							continue;
						}

						if(checkpoint(Point(cal_x , cal_y) , in) && checkpoint(Point(cal_pre_x , cal_pre_y) , in)){//in image scale
							if(CaculatedFeaturePoint.at<uchar>(point_get.y , point_get.x)==1)continue;
							if(fabs((float)in.at<uchar>(cal_y , cal_x)-(float)in.at<uchar>(cal_pre_y , cal_pre_x))<gradient_threshold){
								continue;
							}else{
								
								feature_pre_filter.push_back(point_get);																				
								++count_line_halting[angel_member];
								++count_in;
								CaculatedFeaturePoint.at<uchar>(point_get.y , point_get.x)=1;

								if(count_line_halting[angel_member]>num_per_line-1)
									check_line_halting[angel_member]=true;
							}// end else
						}else{//not in image scale
							check_line_halting[angel_member]=true;
						}
					}// end if		
				}//end for angle

				radius+=radius_stage2_initial;
				for(int k=0;k<num;++k){
					if(check_line_halting[k]==false){
						Finish_stage2 = false;
					}
				}					
			}//end while		
			delete [] check_line_halting;		
			delete [] count_line_halting;		
		}//end for each feature 
		
		double time_start1 = getTickCount();
	
		for(int i=0;i<feature_pre_filter.size();++i){
			sum_x+=feature_pre_filter.at(i).x;
			sum_y+=feature_pre_filter.at(i).y;
		}
		center_x = sum_x/feature_pre_filter.size();
		center_y = sum_y/feature_pre_filter.size();
		

		start_point.x = center_x;
		start_point.y = center_y;

		feature = feature_pre_filter;


		if(DistanceCaculate(start_point ,old_point)<=5)
			break;
		
		++count_cycle;	

		if(count_cycle>10){	
			break;
		}

		delete	[] check_line_halting_part1;
		delete	[] check_line_halting_part2;
		delete  [] count_line_halting_part1;
		delete	[] count_line_halting_part2;	

		time_eye_position_detection_limbus_feature_detection_serial.push_back(getTickCount() - time_start1);
	}//end while	
}

class Parallel_LimbusFeatureDetection : public cv::ParallelLoopBody
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
	Parallel_LimbusFeatureDetection(const Mat &Src, vector<Point> *upperEyelid_feture_arr, vector<Point> *lowerEyelid_feture_arr,
		int *start_loc, int *end_loc, const int &eyeRegionCenter_y, const Mat &Sclera_mask, int diffVal)
		: Src(Src), Sclera_mask(Sclera_mask), upperEyelid_feture_arr(upperEyelid_feture_arr), lowerEyelid_feture_arr(lowerEyelid_feture_arr)
		, start_loc(start_loc), end_loc(end_loc), eyeRegionCenter_y(eyeRegionCenter_y), diff(diffVal) {}

	virtual void operator()(const cv::Range& range) const
	{

		//#pragma omp parallel for		
		for (int i = range.start; i < range.end; ++i)
		{
			/* divide image in 'diff' number
			of parts and process simultaneously */

		}//end for i=range.start
	}
};


