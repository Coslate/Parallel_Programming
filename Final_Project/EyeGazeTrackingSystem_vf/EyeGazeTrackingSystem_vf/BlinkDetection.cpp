#include "BlinkDetection.h"



inline void DetermineBlinking(const bool pre_eyeState , const bool eyeState , int &voluntaryBlinkCount , int &non_voluntaryBlinkCount 
										, int &countCloseFrame){
	if(eyeState==Eyeclose){
		++countCloseFrame;
	}else{
		if(pre_eyeState==Eyeclose){
			if(countCloseFrame<12){
				++non_voluntaryBlinkCount;
			}else{
				if(countCloseFrame<60)
					++voluntaryBlinkCount;
			}
		}
		countCloseFrame = 0;
	}
}

inline void MedianFilter1D(const deque<float> &ratio_queue , float &ratio_return , const int &median_filter1D_size){
	deque<float> ratio_queue_cal(ratio_queue);
	sort(ratio_queue_cal.begin() , ratio_queue_cal.end());
	ratio_return = ratio_queue_cal[median_filter1D_size/2];
}


void EyeBlinkDetection(const bool &noLimbusFeaturePts  , const bool &caculateIris_Mask_done								
									, bool &eyeState , bool &pre_eyeState 
									, int &voluntaryBlinkCount , int &non_voluntaryBlinkCount 
									, int &countCloseFrame
									, const vector<Point> &IrisContoursPoints , const bool &getIrisContourPoints
									, const bool &extremeRight_forBlink , const bool &extremeDown_forBlink
									, const bool &extremeUp_forBlink , const bool &extremeLeft_forBlink
									, deque<float> &ratio_queue
									, const int &irisContour_size
									, const float &eyeCloseDetermine_irisContourSizeThreshold_colorModelBased)
{	
	bool eyeCloseByColor;
	bool eyeCloseByIntensity = noLimbusFeaturePts;
	bool eyeInExtremeRegion = extremeRight_forBlink | extremeDown_forBlink |  extremeUp_forBlink | extremeLeft_forBlink;
	if(eyeInExtremeRegion){
		eyeState = Eyeopen;
		DetermineBlinking(pre_eyeState, eyeState ,voluntaryBlinkCount , non_voluntaryBlinkCount , countCloseFrame);
		pre_eyeState = eyeState;
		return;
	}
	if(caculateIris_Mask_done){		
		if(getIrisContourPoints){					
			if(irisContour_size<eyeCloseDetermine_irisContourSizeThreshold_colorModelBased){
				eyeState = Eyeclose;
				DetermineBlinking(pre_eyeState, eyeState ,voluntaryBlinkCount , non_voluntaryBlinkCount , countCloseFrame);
				pre_eyeState = eyeState;
				return;
			}
			float ratio;

			RotatedRect minRect = minAreaRect(Mat(IrisContoursPoints));
			Point2f rect_points[4]; minRect.points( rect_points );

			if(minRect.size.width>minRect.size.height){
				ratio = minRect.size.width/minRect.size.height;
			}else{
				ratio = minRect.size.height/minRect.size.width;
			}
			
			if(ratio_queue.size()<median_filter1D_size){
				ratio_queue.push_back(ratio);
			}else{
				ratio_queue.pop_front();
				ratio_queue.push_back(ratio);

				MedianFilter1D(ratio_queue , ratio , median_filter1D_size);
			}


			if(ratio>eyeOpenContourRatioThresh_colorModelBased){
				eyeCloseByColor = true;				
			}else{
				eyeCloseByColor = false;			
			}

			if(eyeCloseByColor | eyeCloseByIntensity){
				eyeState = Eyeclose;
			}else{
				eyeState = Eyeopen;
			}
			
		}else{
			eyeState = Eyeclose;
		}
	}else{
		if(noLimbusFeaturePts){
			eyeState = Eyeclose;
		}else{
			eyeState = Eyeopen;
		}
	}

	DetermineBlinking(pre_eyeState, eyeState ,voluntaryBlinkCount , non_voluntaryBlinkCount , countCloseFrame);
	pre_eyeState = eyeState;
}

