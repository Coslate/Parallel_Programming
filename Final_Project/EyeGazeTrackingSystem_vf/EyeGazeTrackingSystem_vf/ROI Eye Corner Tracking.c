	//-------------------ROI Eye Corner Tracking--------------------//	
	//if(frame_number==/*305*/181){
	//	if(!(RefineEyeCornerROI( roi_LeftWidth , roi_LeftHeight ,roi_RightWidth,roi_RightHeight
	//											 , rectLeftCornerStartPoint, rectRightCornerStartPoint
	//											 , leftEyeCorner , rightEyeCorner
	//											 ,leftCorrect_x_left , leftCorrect_x_right , leftCorrect_y_up , leftCorrect_y_down
	//											 ,rightCorrect_x_left ,rightCorrect_x_right , rightCorrect_y_up , rightCorrect_y_down
	//											 ,width_LeftROIforFollowingTM ,height_LeftROIforFollowingTM
	//											 ,width_RightROIforFollowingTM ,height_RightROIforFollowingTM)))return;
	//
	//	
	//	region_of_interestLeftCorner = Rect(rectLeftCornerStartPoint.x, rectLeftCornerStartPoint.y
	//													, width_LeftROIforFollowingTM, height_LeftROIforFollowingTM);
	//	region_of_interestRightCorner = Rect(rectRightCornerStartPoint.x, rectRightCornerStartPoint.y
	//													, width_RightROIforFollowingTM, height_RightROIforFollowingTM);
	//
	//	ROI_Leftcorner = Frame_Gray(region_of_interestLeftCorner);
	//	ROI_Rightcorner = Frame_Gray(region_of_interestRightCorner);
	//}
	//	
	//rectangle(Frame, rectLeftCornerStartPoint, rectLeftCornerStartPoint+Point(width_LeftROIforFollowingTM , height_LeftROIforFollowingTM)
	//				, Scalar(0,200 , 255) , 3);

	//rectangle(Frame, rectRightCornerStartPoint, rectRightCornerStartPoint+Point(width_RightROIforFollowingTM , height_RightROIforFollowingTM)
	//				, Scalar(0,200 , 255) , 3);


	//Rect ROI_LeftimageFrameGray = Rect(0 , 0 , Frame_Gray.cols/2 , Frame.rows);
	//Rect ROI_RightimageFrameGray = Rect(Frame_Gray.cols/2 , 0 , Frame_Gray.cols/2 , Frame.rows);
	//Mat ROI_LeftImageGray = Frame_Gray(ROI_LeftimageFrameGray);
	//Mat ROI_RightImageGray = Frame_Gray(ROI_RightimageFrameGray);
	//
	//
	//ROIEyeCornerTracking(ROI_Leftcorner, ROI_Rightcorner , rectLeftCornerStartPoint, rectRightCornerStartPoint
	//									, leftEyeCorner	, rightEyeCorner , ROI_LeftImageGray , ROI_RightImageGray);	
	//
	////cout<<"========frame_num_"<<frame_number<<"========="<<endl;
	//leftEyeCorner = leftEyeCorner+Point(leftCorrect_x_left , leftCorrect_y_up);
	//rightEyeCorner = rightEyeCorner+Point(rightCorrect_x_left , rightCorrect_y_up)+Point(Frame_Gray.cols/2 , 0);
	////cout<<"leftEyeCorner = "<<leftEyeCorner<<endl;
	////cout<<"Point(leftCorrect_x_left , leftCorrect_y_up) = "<<Point(leftCorrect_x_left , leftCorrect_y_up)<<endl;

	////cout<<"norm(leftCornerOriginalPoint - leftEyeCorner) = "<<norm(leftCornerOriginalPoint - leftEyeCorner)<<endl;

	//if(norm(leftCornerOriginalPoint - leftEyeCorner)>100){
	//	leftEyeCorner = leftCornerOriginalPoint;
	//}
	//if(norm(rightCornerOriginalPoint -rightEyeCorner)>100){
	//	rightEyeCorner = rightCornerOriginalPoint;
	//}


	//
	//if(!(RefineEyeCornerROI( roi_LeftWidth , roi_LeftHeight ,roi_RightWidth,roi_RightHeight
	//											, rectLeftCornerStartPoint, rectRightCornerStartPoint
	//											, leftEyeCorner , rightEyeCorner
	//											,leftCorrect_x_left , leftCorrect_x_right , leftCorrect_y_up , leftCorrect_y_down
	//											,rightCorrect_x_left ,rightCorrect_x_right , rightCorrect_y_up , rightCorrect_y_down
	//											,width_LeftROIforFollowingTM ,height_LeftROIforFollowingTM
	//											,width_RightROIforFollowingTM ,height_RightROIforFollowingTM)))return;
	//

	//region_of_interestLeftCorner = Rect(rectLeftCornerStartPoint.x, rectLeftCornerStartPoint.y
	//												, width_LeftROIforFollowingTM, height_LeftROIforFollowingTM);
	//region_of_interestRightCorner = Rect(rectRightCornerStartPoint.x, rectRightCornerStartPoint.y
	//												, width_RightROIforFollowingTM, height_RightROIforFollowingTM);
	//
	//ROI_Leftcorner = Frame_Gray(region_of_interestLeftCorner);
	//ROI_Rightcorner = Frame_Gray(region_of_interestRightCorner);




