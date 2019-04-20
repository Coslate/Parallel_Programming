#include "MappingFunction.h"


inline Point2d* normalize_edge_pointSet(double &dis_scale, Point2d &nor_center, const int &ep_num 
																, const vector<Point> &feature_point)
{
	const float sqrt_2 = 1.414213;
	double sumx = 0, sumy = 0;
	double sumdis = 0;
	Point edge;
	Point original(0, 0);

	//#pragma omp parallel for 
	for (int i = 0; i < ep_num; ++i){
		edge = feature_point.at(i);
		sumx += edge.x;
		sumy += edge.y;
		sumdis += sqrtf((float)(edge.x*edge.x + edge.y*edge.y));
		//sumdis+=DistanceCaculate(edge,  original);
	}

	dis_scale = sqrt_2*ep_num/sumdis;
	nor_center.x = sumx*1.0/ep_num;
	nor_center.y = sumy*1.0/ep_num;
	Point2d *edge_point_nor = new Point2d[ep_num];
	for (int i = 0; i < ep_num; ++i){
		edge = feature_point.at(i);
		edge_point_nor[i].x = ((float)edge.x - nor_center.x)*dis_scale;
		edge_point_nor[i].y = ((float)edge.y - nor_center.y)*dis_scale;		
	}
	return edge_point_nor;
}

inline Point2d* normalize_edge_pointSetRevise(double &dis_scale, Point2d &nor_center, const int &ep_num 
																, const vector<Point> &feature_point)
{
	const float sqrt_2 = 1.414213;
	double sumx = 0, sumy = 0;
	double sumdis = 0;
	Point edge;
	Point original(0, 0);

	//#pragma omp parallel for 
	for (int i = 0; i < ep_num; ++i){		
		sumx += feature_point[i].x;
		sumy += feature_point[i].y;
		//sumdis += sqrtf((float)(edge.x*edge.x + edge.y*edge.y));
		//sumdis+=DistanceCaculate(edge,  original);
	}
	
	nor_center.x = sumx/(double)ep_num;
	nor_center.y = sumy/(double)ep_num;
	Point2d *edge_point_nor = new Point2d[ep_num];
	for (int i = 0; i < ep_num; ++i){		
		edge_point_nor[i].x = ((double)feature_point[i].x - nor_center.x);
		edge_point_nor[i].y = ((double)feature_point[i].y - nor_center.y);		
		sumdis += sqrtf((double)(edge_point_nor[i].x*edge_point_nor[i].x + edge_point_nor[i].y*edge_point_nor[i].y));
	}

	dis_scale = sqrt_2*ep_num/sumdis;
	for(int i=0;i<ep_num;++i){
		edge_point_nor[i].x = ((double)feature_point[i].x - nor_center.x)*dis_scale;
		edge_point_nor[i].y = ((double)feature_point[i].y - nor_center.y)*dis_scale;		
	}
	return edge_point_nor;
}


inline bool MappingEyeToGaze_X(double* &mapping_paramX , const int &numberOfVar , const int &n_order 			
												, const vector<Point> &scenePoints_Set_in , const vector<Point> &eyePoints_Set_in																					
												, const Point2d * const eye_nor , const Point2d* const scene_nor
												, Mat &A_CoeffMatrix)
{
	mapping_paramX = new double [numberOfVar]();
	if(scenePoints_Set_in.empty() || eyePoints_Set_in.empty()){
		printf("There is no calibration points in MappingEyeToGaze_X()!\n");
		return false;
	}else if(scenePoints_Set_in.size()<numberOfVar  || eyePoints_Set_in.size()<numberOfVar){
		printf("There is no enough calibration points in MappingEyeToGaze_X()!\n");
		return false;
	}
	
	//Matrix Construction [A][Coeff] = [0]	
	int M = numberOfVar;
	int N = numberOfVar;	
	A_CoeffMatrix = Mat::zeros(M , N , CV_64F);
	Mat sceneMatrix(M , 1 , CV_64F);	
	Mat W , U , Vt;
	Mat Ut , V , W_inv;
	Mat result;
	SVD svd_opencv;

	for(int i=0;i<A_CoeffMatrix.rows;++i){
		A_CoeffMatrix.at<double>(i , A_CoeffMatrix.cols-1) = 1;
	}

	for (int i=0; i<A_CoeffMatrix.rows; ++i){
		int pos_horiz = 0;
		for(int p=1;p<n_order+1;++p){
			for(int k=0;k<p+1;++k){
				A_CoeffMatrix.at<double>(i , pos_horiz) = powf(eye_nor[i].x , p-k)*powf(eye_nor[i].y , k);
				++pos_horiz;
			}
		}		
	}

	for(int i=0;i<sceneMatrix.rows;++i){
		sceneMatrix.at<double>(i , 0) = scene_nor[i].x;
	}

	//SVD	
	svd_opencv.compute(A_CoeffMatrix, W , U, Vt, SVD::FULL_UV);		

	
	//Caculate result = V*W_inv*Ut*sceneMatrix
	Mat wMat = Mat::zeros(M, N , CV_64F);
	for(int i=0;i<M;++i){
		for(int j=0;j<N;++j){
			if(i==j){
				wMat.at<double>(i , j) = W.at<double>(i);
			}
		}
	}
	W_inv = wMat.inv();
	transpose(U , Ut);
	transpose(Vt , V);
	result = V*W_inv*Ut*sceneMatrix;
	
	for(int i=0;i<numberOfVar;++i){
		mapping_paramX[i] = result.at<double>(i , 0);
	}

	return true;
}


inline bool MappingEyeToGaze_Y(double* &mapping_paramY , const int &numberOfVar , const int &n_order 			
												, const vector<Point> &scenePoints_Set_in , const vector<Point> &eyePoints_Set_in																					
												, const Point2d * const eye_nor , const Point2d* const scene_nor
												, const Mat &A_CoeffMatrix)
{
	mapping_paramY = new double [numberOfVar]();
	if(scenePoints_Set_in.empty() || eyePoints_Set_in.empty()){
		printf("There is no calibration points in MappingEyeToGaze_Y()!\n");
		return false;
	}else if(scenePoints_Set_in.size()<numberOfVar  || eyePoints_Set_in.size()<numberOfVar){
		printf("There is no enough calibration points in MappingEyeToGaze_Y()!\n");
		return false;
	}
	
	//Matrix Construction [A][Coeff] = [0]
	int M = numberOfVar;
	int N = numberOfVar;	
	Mat sceneMatrix(M , 1 , CV_64F);	
	Mat W , U , Vt;
	Mat Ut , V , W_inv;
	Mat result;
	SVD svd_opencv;

	for(int i=0;i<sceneMatrix.rows;++i){
		sceneMatrix.at<double>(i , 0) = scene_nor[i].y;
	}

	//SVD		
	svd_opencv.compute(A_CoeffMatrix, W , U, Vt, SVD::FULL_UV);	


	//Caculate result = V*W_inv*Ut*sceneMatrix
	Mat wMat = Mat::zeros(M, N , CV_64F);
	for(int i=0;i<M;++i){
		for(int j=0;j<N;++j){
			if(i==j){
				wMat.at<double>(i , j) = W.at<double>(i);
			}
		}
	}
	
	W_inv = wMat.inv();	
	transpose(U , Ut);
	transpose(Vt , V);
	result = V*W_inv*Ut*sceneMatrix;
	
	for(int i=0;i<numberOfVar;++i){
		mapping_paramY[i] = result.at<double>(i , 0);
	}

	return true;
}

inline bool MappingEyeToGaze_X_ALLCaculated(double* &mapping_paramX , const int &numberOfVar , const int &n_order 			
																	, const vector<Point> &scenePoints_Set_in , const vector<Point> &eyePoints_Set_in																					
																	, const Point2d * const eye_nor , const Point2d* const scene_nor
																	, Mat &A_CoeffMatrix)
{
	mapping_paramX = new double [numberOfVar]();
	if(scenePoints_Set_in.empty() || eyePoints_Set_in.empty()){
		printf("There is no calibration points in MappingEyeToGaze_X()!\n");
		return false;
	}else if(scenePoints_Set_in.size()<numberOfVar  || eyePoints_Set_in.size()<numberOfVar){
		printf("There is no enough calibration points in MappingEyeToGaze_X()!\n");
		return false;
	}
	
	//Matrix Construction [A][Coeff] = [0]	
	int M = scenePoints_Set_in.size();
	int N = numberOfVar;	
	A_CoeffMatrix = Mat::zeros(M , N , CV_64F);
	Mat sceneMatrix(M , 1 , CV_64F);	
	Mat W , U , Vt;
	Mat Ut , V , W_inv;
	Mat result;
	SVD svd_opencv;

	for(int i=0;i<A_CoeffMatrix.rows;++i){
		A_CoeffMatrix.at<double>(i , A_CoeffMatrix.cols-1) = 1;
	}
	
	for (int i=0; i<A_CoeffMatrix.rows; ++i){
		int pos_horiz = 0;
		for(int p=1;p<n_order+1;++p){
			for(int k=0;k<p+1;++k){
				A_CoeffMatrix.at<double>(i , pos_horiz) = powf(eye_nor[i].x , p-k)*powf(eye_nor[i].y , k);
				++pos_horiz;
			}
		}		
	}
	
	for(int i=0;i<sceneMatrix.rows;++i){
		sceneMatrix.at<double>(i , 0) = scene_nor[i].x;
	}

	//SVD	
	svd_opencv.compute(A_CoeffMatrix, W , U, Vt, SVD::FULL_UV);		

	
	//Caculate result = V*W_inv*Ut*sceneMatrix
	Mat wMat = Mat::zeros(M, N , CV_64F);
	for(int i=0;i<M;++i){
		for(int j=0;j<N;++j){
			if(i==j){
				wMat.at<double>(i , j) = W.at<double>(i);
			}
		}
	}
	
	W_inv = wMat.inv(DECOMP_SVD);
	transpose(U , Ut);
	transpose(Vt , V);
	result = V*W_inv*Ut*sceneMatrix;
		
	for(int i=0;i<numberOfVar;++i){
		mapping_paramX[i] = result.at<double>(i , 0);
	}

	return true;
}

inline bool MappingEyeToGaze_Y_ALLCaculated(double* &mapping_paramY , const int &numberOfVar , const int &n_order 			
																		, const vector<Point> &scenePoints_Set_in , const vector<Point> &eyePoints_Set_in																					
																		, const Point2d * const eye_nor , const Point2d* const scene_nor
																		, const Mat &A_CoeffMatrix)
{
	mapping_paramY = new double [numberOfVar]();
	if(scenePoints_Set_in.empty() || eyePoints_Set_in.empty()){
		printf("There is no calibration points in MappingEyeToGaze_Y()!\n");
		return false;
	}else if(scenePoints_Set_in.size()<numberOfVar  || eyePoints_Set_in.size()<numberOfVar){
		printf("There is no enough calibration points in MappingEyeToGaze_Y()!\n");
		return false;
	}
	
	//Matrix Construction [A][Coeff] = [0]
	int M = scenePoints_Set_in.size();
	int N = numberOfVar;	
	Mat sceneMatrix(M , 1 , CV_64F);	
	Mat W , U , Vt;
	Mat Ut , V , W_inv;
	Mat result;
	SVD svd_opencv;

	for(int i=0;i<sceneMatrix.rows;++i){
		sceneMatrix.at<double>(i , 0) = scene_nor[i].y;
	}

	//SVD		
	svd_opencv.compute(A_CoeffMatrix, W , U, Vt, SVD::FULL_UV);	


	//Caculate result = V*W_inv*Ut*sceneMatrix
	Mat wMat = Mat::zeros(M, N , CV_64F);
	for(int i=0;i<M;++i){
		for(int j=0;j<N;++j){
			if(i==j){
				wMat.at<double>(i , j) = W.at<double>(i);
			}
		}
	}
	
	W_inv = wMat.inv(DECOMP_SVD);	
	transpose(U , Ut);
	transpose(Vt , V);
	result = V*W_inv*Ut*sceneMatrix;
	
	for(int i=0;i<numberOfVar;++i){
		mapping_paramY[i] = result.at<double>(i , 0);
	}

	return true;
}

inline bool get_randomPts_Num(const int &max_num , const int &numofPts , int* &rand_num){
	int rand_index = 0;
	int r;	
	bool is_new = 1;
	rand_num = new int [numofPts]();

	srand(time(NULL));
	if(max_num<numofPts-1){
		printf("Doesn't get enough eye-gaze points pairs for calibration precedure.\n");
		return false;
	}

	if (max_num == numofPts-1) {
		for (int i = 0; i < numofPts; ++i) {
			rand_num[i] = i;
		}
		return true;
	}

	while (rand_index < numofPts){
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

inline void EstimateErrorOfModel(const vector<Point> &calibratedEyeRefinedCenter 
												, const vector<Point> &calibratedCalPoints
												, const Mat &EyePtsTransformMat
												, const Mat &ScenePtsTransformMat
												, const double *const mapping_paramX , const double *const mapping_paramY
												, const int &numberOfVar , const int &n_order
												, double &meanSquareError)
{
	meanSquareError = 0;
	for(int i=0;i<calibratedCalPoints.size();++i){
		double est_x = 0;
		double est_y = 0;

		Mat eyeNormalzed(3 , 1 , CV_64F);
		Mat eyeOriginal(3 , 1 , CV_64F);
		Mat SceneNormalized(3 , 1 , CV_64F);
		Mat SceneDeNormalized(3 , 1 , CV_64F);
		Mat A_CoeffMatrix(1 , numberOfVar , CV_64F);

		eyeOriginal.at<double>(0 , 0) = calibratedEyeRefinedCenter[i].x;
		eyeOriginal.at<double>(1 , 0) = calibratedEyeRefinedCenter[i].y;
		eyeOriginal.at<double>(2 , 0) = 1;		
		eyeNormalzed = EyePtsTransformMat*eyeOriginal;
				
		int pos_horiz = 0;
		for(int p=1;p<n_order+1;++p){
			for(int k=0;k<p+1;++k){				
				A_CoeffMatrix.at<double>(0 , pos_horiz) = powf(eyeNormalzed.at<double>(0 , 0) , p-k)*powf(eyeNormalzed.at<double>(1 , 0) , k);				
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
		
		
		SceneDeNormalized = ScenePtsTransformMat.inv()*SceneNormalized;
		
		meanSquareError += sqrtf(powf(SceneDeNormalized.at<double>(0 , 0)/SceneDeNormalized.at<double>(2 , 0) - calibratedCalPoints[i].x , 2) 
										+ powf(SceneDeNormalized.at<double>(1 , 0)/SceneDeNormalized.at<double>(2 , 0) - calibratedCalPoints[i].y , 2));
	}
	meanSquareError/=calibratedCalPoints.size();
}

inline double NumFactorial(const int &n){
	double sum = 1;
	if(n==0 || n==1){
		sum = 1;
	}else{
		for(int i=n;i>0;--i){
			sum*=i;
		}
	}
	return sum;
}

bool MappingEyeGaze_PolyNomialRANSAC(const std::vector<Point> &calibratedEyeRefinedCenter 
																				, const std::vector<Point> &calibratedCalPoints
																				, double* &mapping_paramOptX , double* &mapping_paramOptY 
																				, const int &mappingOrder , int &numberOfVar
																				, Mat &EyePtsTransformMat_Opt
																				, Mat &ScenePtsTransformMat_Opt)
{	
	EyePtsTransformMat_Opt = Mat::zeros(3 , 3 , CV_64F);
	ScenePtsTransformMat_Opt = Mat::zeros(3 , 3 , CV_64F);
	int numOfAllCalPts = calibratedCalPoints.size();
	double minError = DBL_MAX;
	int countRANSAC_times = 0;
	bool mappingDone = true;
	bool mappingX = false;
	bool mappingY = false;
	numberOfVar = (mappingOrder+3)*mappingOrder/2+1;


	int sample_num = NumFactorial(calibratedCalPoints.size())/(NumFactorial(numberOfVar)*NumFactorial(calibratedCalPoints.size() - numberOfVar));		
	mapping_paramOptX = new double [numberOfVar]();
	mapping_paramOptY = new double [numberOfVar]();
	
	while(countRANSAC_times<sample_num){
		int *rand_num;
		Point2d scene_center, eye_center, *eye_nor, *scene_nor;
		double dis_scale_scene, dis_scale_eye;  
		double *mapping_paramX;
		double *mapping_paramY;
		double meanSquareError;
		std::vector<Point> scenePoints_Set_in;
		std::vector<Point> eyePoints_Set_in;
		Mat A_CoeffMatrix;
		Mat EyePtsTransformMat = Mat::zeros(3 , 3 , CV_64F);
		Mat ScenePtsTransformMat = Mat::zeros(3 , 3 , CV_64F);

		//Get random numberOfVar pts
		if(!get_randomPts_Num(numOfAllCalPts - 1 , numberOfVar , rand_num)){
			mappingDone = false;
			break;
		}

		//Normalize
		for(int i=0;i<numberOfVar;++i){
			scenePoints_Set_in.push_back(calibratedCalPoints[rand_num[i]]);
			eyePoints_Set_in.push_back(calibratedEyeRefinedCenter[rand_num[i]]);
		}
		scene_nor = normalize_edge_pointSet(dis_scale_scene, scene_center, numberOfVar , scenePoints_Set_in);
		eye_nor = normalize_edge_pointSet(dis_scale_eye, eye_center, numberOfVar , eyePoints_Set_in);		
		

		//Forming transform matrix
		EyePtsTransformMat.at<double>(0 , 0) = dis_scale_eye;
		EyePtsTransformMat.at<double>(1 , 1) = dis_scale_eye;
		EyePtsTransformMat.at<double>(0 , 2) = -dis_scale_eye*eye_center.x;
		EyePtsTransformMat.at<double>(1 , 2) = -dis_scale_eye*eye_center.y;
		EyePtsTransformMat.at<double>(2 , 2) = 1;		

		ScenePtsTransformMat.at<double>(0 , 0) = dis_scale_scene;
		ScenePtsTransformMat.at<double>(1 , 1) = dis_scale_scene;
		ScenePtsTransformMat.at<double>(0 , 2) = -dis_scale_scene*scene_center.x;
		ScenePtsTransformMat.at<double>(1 , 2) = -dis_scale_scene*scene_center.y;
		ScenePtsTransformMat.at<double>(2 , 2) = 1;		

		//Caculate mapping function			
		mappingX = MappingEyeToGaze_X(mapping_paramX , numberOfVar , mappingOrder 			
											, scenePoints_Set_in , eyePoints_Set_in																					
											, eye_nor , scene_nor
											, A_CoeffMatrix);
	
		mappingY = MappingEyeToGaze_Y(mapping_paramY , numberOfVar , mappingOrder
											, scenePoints_Set_in , eyePoints_Set_in																					
											, eye_nor , scene_nor
											, A_CoeffMatrix);
		
		if(!(mappingX&mappingY)){
			mappingDone = false;
			break;
		}
		//Estimate error of the model		
		EstimateErrorOfModel(calibratedEyeRefinedCenter 
											, calibratedCalPoints
											, EyePtsTransformMat
											, ScenePtsTransformMat
											, mapping_paramX , mapping_paramY
											, numberOfVar , mappingOrder
											, meanSquareError);


		if(meanSquareError<minError){
			minError = meanSquareError;
			for(int i=0;i<numberOfVar;++i){
				mapping_paramOptX[i] = mapping_paramX[i];
				mapping_paramOptY[i] = mapping_paramY[i];
			}
			EyePtsTransformMat_Opt = EyePtsTransformMat.clone();
			ScenePtsTransformMat_Opt = ScenePtsTransformMat.clone();
		}
		++countRANSAC_times;
		if(countRANSAC_times>3000)break;

	}//end while
	if(mappingDone){
		cout<<"minError = "<<minError<<endl;
	}
	
	return mappingDone;
}


bool MappingEyeGaze_PolyNomialALLPtsCaculated(const std::vector<Point> &calibratedEyeRefinedCenter 
																					, const std::vector<Point> &calibratedCalPoints
																					, double* &mapping_paramOptX , double* &mapping_paramOptY 
																					, const int &mappingOrder , int &numberOfVar
																					, Mat &EyePtsTransformMat_Opt
																					, Mat &ScenePtsTransformMat_Opt
																					, double &meanSquareError)
{
	EyePtsTransformMat_Opt = Mat::zeros(3 , 3 , CV_64F);
	ScenePtsTransformMat_Opt = Mat::zeros(3 , 3 , CV_64F);
	int numOfAllCalPts = calibratedCalPoints.size();
	bool mappingDone = true;
	bool mappingX = false;
	bool mappingY = false;
	Point2d scene_center, eye_center, *eye_nor, *scene_nor;
	double dis_scale_scene, dis_scale_eye;  		
	std::vector<Point> scenePoints_Set_in;
	std::vector<Point> eyePoints_Set_in;
	Mat A_CoeffMatrix;	
	numberOfVar = (mappingOrder+3)*mappingOrder/2+1;


	//Normalize
	for(int i=0;i<calibratedCalPoints.size();++i){
		scenePoints_Set_in.push_back(calibratedCalPoints[i]);
		eyePoints_Set_in.push_back(calibratedEyeRefinedCenter[i]);
	}
	scene_nor = normalize_edge_pointSet(dis_scale_scene, scene_center, calibratedCalPoints.size(), scenePoints_Set_in);
	eye_nor = normalize_edge_pointSet(dis_scale_eye, eye_center, calibratedCalPoints.size() , eyePoints_Set_in);
	
		
	//Forming transform matrix
	EyePtsTransformMat_Opt.at<double>(0 , 0) = dis_scale_eye;
	EyePtsTransformMat_Opt.at<double>(1 , 1) = dis_scale_eye;
	EyePtsTransformMat_Opt.at<double>(0 , 2) = -dis_scale_eye*eye_center.x;
	EyePtsTransformMat_Opt.at<double>(1 , 2) = -dis_scale_eye*eye_center.y;
	EyePtsTransformMat_Opt.at<double>(2 , 2) = 1;		

	ScenePtsTransformMat_Opt.at<double>(0 , 0) = dis_scale_scene;
	ScenePtsTransformMat_Opt.at<double>(1 , 1) = dis_scale_scene;
	ScenePtsTransformMat_Opt.at<double>(0 , 2) = -dis_scale_scene*scene_center.x;
	ScenePtsTransformMat_Opt.at<double>(1 , 2) = -dis_scale_scene*scene_center.y;
	ScenePtsTransformMat_Opt.at<double>(2 , 2) = 1;		

	//Caculate mapping function			
	mappingX = MappingEyeToGaze_X_ALLCaculated(mapping_paramOptX , numberOfVar , mappingOrder 			
																				, scenePoints_Set_in , eyePoints_Set_in																					
																				, eye_nor , scene_nor
																				, A_CoeffMatrix);
	
	mappingY = MappingEyeToGaze_Y_ALLCaculated(mapping_paramOptY , numberOfVar , mappingOrder
																				, scenePoints_Set_in , eyePoints_Set_in																					
																				, eye_nor , scene_nor
																				, A_CoeffMatrix);
		
	if(!(mappingX&mappingY)){
		mappingDone = false;		
	}
	//Estimate error of the model		
	if(mappingDone){
		EstimateErrorOfModel(calibratedEyeRefinedCenter 
										, calibratedCalPoints
										, EyePtsTransformMat_Opt
										, ScenePtsTransformMat_Opt
										, mapping_paramOptX , mapping_paramOptY
										, numberOfVar , mappingOrder
										, meanSquareError);		
	}
		
	return mappingDone;
}

bool MappingEyeGaze_PolyNomialALLOrderCaculated(const std::vector<Point> &calibratedEyeRefinedCenter 
																					, const std::vector<Point> &calibratedCalPoints
																					, double* &mapping_paramOptX , double* &mapping_paramOptY 
																					, int &numberOfVarOpt
																					, Mat &EyePtsTransformMat_Opt
																					, Mat &ScenePtsTransformMat_Opt
																					, const int &calPtsLength
																					, int &orderOpt
																					, int &calibrationPts_space)
{
	int numOfCalPts;
	bool stopInputOrder = false;
	int orderCount = 0;
	vector<int> possibleOrder;
	double minErr = FLT_MAX;
	vector<double> collectMapping_paramX;
	vector<double> collectMapping_paramY;

	if(calibrationPts_space==calibrationPattern::Step_space_two){
		if(calPtsLength%2==0){
			numOfCalPts = calPtsLength/2*calPtsLength;
		}else{
			int qUse = calPtsLength/2;
			int rUse = 1;
			numOfCalPts = calPtsLength*qUse + ceil(calPtsLength/2);
		}	
	}else{
		numOfCalPts = calPtsLength*calPtsLength;
	}

	while(!stopInputOrder){
		++orderCount;
		if((orderCount+3)*orderCount/2 + 1<=numOfCalPts){
			possibleOrder.push_back(orderCount);
		}else{
			stopInputOrder = true;
		}		
	}

	ofstream fileOut(".//PolyOrder.txt");

	for(int i=0;i<possibleOrder.size();++i){
		int numberOfVar;
		Mat EyePtsTransformMat;
		Mat ScenePtsTransformMat;
		double meanSquareError;
		double *mapping_paramX;
		double *mapping_paramY;

		fileOut<<"order = "<<possibleOrder[i]<<" : "<<endl;

		if(MappingEyeGaze_PolyNomialALLPtsCaculated(calibratedEyeRefinedCenter 
																				, calibratedCalPoints
																				, mapping_paramX , mapping_paramY 
																				, possibleOrder[i] , numberOfVar
																				, EyePtsTransformMat
																				, ScenePtsTransformMat
																				, meanSquareError))
		{
			fileOut<<"numberOfVar = "<<numberOfVar<<endl;
			fileOut<<endl;
			fileOut<<"mapping_paramX = "<<endl;
			for(int j=0;j<numberOfVar;++j){
				fileOut<<mapping_paramX[j]<<" ";
			}
			fileOut<<endl;
			fileOut<<"mapping_paramY = "<<endl;
			for(int j=0;j<numberOfVar;++j){
				fileOut<<mapping_paramY[j]<<" ";
			}
			fileOut<<endl;
			fileOut<<"EyePtsTransformMat = "<<endl;
			for(int j=0;j<EyePtsTransformMat.rows;++j){
				for(int k=0;k<EyePtsTransformMat.cols;++k){
					fileOut<<EyePtsTransformMat.at<double>(j , k)<<" ";
				}
				fileOut<<endl;
			}
			fileOut<<endl;
			fileOut<<"ScenePtsTransformMat = "<<endl;
			for(int j=0;j<ScenePtsTransformMat.rows;++j){
				for(int k=0;k<ScenePtsTransformMat.cols;++k){
					fileOut<<ScenePtsTransformMat.at<double>(j , k)<<" ";
				}
				fileOut<<endl;
			}
			fileOut<<"meanSquareError = "<<meanSquareError<<endl;
			fileOut<<"================================="<<endl;


			if(meanSquareError<minErr){
				collectMapping_paramX.clear();
				collectMapping_paramY.clear();
				orderOpt = possibleOrder[i];
				numberOfVarOpt = numberOfVar;
				EyePtsTransformMat_Opt = EyePtsTransformMat.clone();
				ScenePtsTransformMat_Opt = ScenePtsTransformMat.clone();
				for(int k=0;k<numberOfVarOpt;++k){
					collectMapping_paramX.push_back(mapping_paramX[k]);
					collectMapping_paramY.push_back(mapping_paramY[k]);
				}	
			}

		}
	}

	mapping_paramOptX = new double[numberOfVarOpt]();
	mapping_paramOptY = new double[numberOfVarOpt]();
	for(int i=0;i<numberOfVarOpt;++i){
		mapping_paramOptX[i] = collectMapping_paramX[i];
		mapping_paramOptY[i] = collectMapping_paramY[i];
	}	

	return true;
}


//bool MappingEyeGaze_SVR(const std::vector<Point> &calibratedEyeRefinedCenter 
//											, const std::vector<Point> &calibratedCalPoints
//											, dlib::decision_function<kernel_type> &svr_model_X
//											, dlib::decision_function<kernel_type> &svr_model_Y)
//{	
//    std::vector<sample_type> samples;  
//	std::vector<double> targets_x;
//	std::vector<double> targets_y;
//
//    //Prepare training set
//    sample_type m;
//	for(int i=0;i<calibratedCalPoints.size();++i){
//		targets_x.push_back(calibratedCalPoints[i].x);
//		targets_y.push_back(calibratedCalPoints[i].y);
//	}
//	for(int i=0;i<calibratedEyeRefinedCenter.size();++i){
//		m(0 , 0) = calibratedEyeRefinedCenter[i].x;
//		m(1 , 0) = calibratedEyeRefinedCenter[i].y;
//		samples.push_back(m);
//	}
//
//	// Now setup a SVR trainer object.  It has three parameters, the kernel and
//    // two parameters specific to SVR.  	
//    dlib::svr_trainer<kernel_type> trainer;
//    trainer.set_kernel(kernel_type(0.01));
//
//    // This parameter is the usual regularization parameter.  It determines the trade-off 
//    // between trying to reduce the training error or allowing more errors but hopefully 
//    // improving the generalization of the resulting function.  Larger values encourage exact 
//    // fitting while smaller values of C may encourage better generalization.
//    trainer.set_c(10);
//
//    // Epsilon-insensitive regression means we do regression but stop trying to fit a data 
//    // point once it is "close enough" to its target value.  This parameter is the value that 
//    // controls what we mean by "close enough".  In this case, I'm saying I'm happy if the
//    // resulting regression function gets within 0.001 of the target value.
//    trainer.set_epsilon_insensitivity(0.001);
//
//    // Now do the training and save the results
//    svr_model_X = trainer.train(samples, targets_x);
//	svr_model_Y = trainer.train(samples, targets_y);
//
//	return true;
//}


bool MappingEyeGaze_HomographySlice(std::vector<sliceMapElement> &mappingSliceMap
																		, const std::vector<Point> &calibratedCalPoints)
{
	bool constructDone = true;	
	for(int i=0;i<mappingSliceMap.size();++i){		
		if(!mappingSliceMap[i].mappingHomographyConstruction(calibratedCalPoints)){			
			printf("\nHomography model of %d model in mappingSliceMap failed.\n" , i);
			return false;	
		}
	}	
	return constructDone;
}



//----------------sliceMapElement function implementation-----------------//
int sliceMapElement::testOut = 0;
inline bool SolveHomography(const Point2d * const scene_nor , const Point2d *const eye_nor
											, const int &correspPtsNum 
											, const Mat &EyePtsTransformMat_Opt
											, const Mat &ScenePtsTransformMat_Opt
											, Mat &Map_Matrix_return)
{		
	const int valNum = 8;
	int M = correspPtsNum*2;
	int N = valNum;
	Mat Map_matrix(3 , 3 , CV_64F);
	Mat A_CoeffMatrix = Mat::zeros(M , N , CV_64F);
	Mat sceneMatrix(M , 1 , CV_64F);	
	Mat W , U , Vt;
	Mat Ut , V , W_inv;
	Mat Result;
	SVD svd_opencv;

	if(M<N){
		printf("\n Don't have enough correspondence through calibration. \n");
		return false;
	}
	
	
	//Forming A Matrix , Ax = b
	for(int j = 0;  j < correspPtsNum; ++j){
		int i = 2 * j;
		A_CoeffMatrix.at<double>(i, 0)
		  = A_CoeffMatrix.at<double>(i, 1)
		  = A_CoeffMatrix.at<double>(i, 2) = 0.0f;
		A_CoeffMatrix.at<double>(i, 3) = eye_nor[j].x;
		A_CoeffMatrix.at<double>(i, 4) = eye_nor[j].y;
		A_CoeffMatrix.at<double>(i, 5) = 1.0f;
		A_CoeffMatrix.at<double>(i, 6) = -scene_nor[j].y * eye_nor[j].x;
		A_CoeffMatrix.at<double>(i, 7) = -scene_nor[j].y * eye_nor[j].y;

		A_CoeffMatrix.at<double>(i + 1, 0) = eye_nor[j].x;
		A_CoeffMatrix.at<double>(i + 1, 1) = eye_nor[j].y;
		A_CoeffMatrix.at<double>(i + 1, 2) = 1;
		A_CoeffMatrix.at<double>(i + 1, 3)
		  = A_CoeffMatrix.at<double>(i + 1, 4)
		  = A_CoeffMatrix.at<double>(i + 1, 5) = 0.0f;
		A_CoeffMatrix.at<double>(i + 1, 6) = -scene_nor[j].x * eye_nor[j].x;
		A_CoeffMatrix.at<double>(i + 1, 7) = -scene_nor[j].x * eye_nor[j].y;

		sceneMatrix.at<double>(i) = scene_nor[j].y;
		sceneMatrix.at<double>(i + 1) = scene_nor[j].x;
	}
	
	//SVD	
	svd_opencv.compute(A_CoeffMatrix, W , U, Vt, SVD::FULL_UV);		

		
	//Caculate result = V*W_inv*Ut*sceneMatrix
	Mat wMat = Mat::zeros(M, N , CV_64F);
	for(int i=0;i<M;++i){
		for(int j=0;j<N;++j){
			if(i==j){
				wMat.at<double>(i , j) = W.at<double>(i);
			}
		}
	}
		
	W_inv = wMat.inv(DECOMP_SVD);
	transpose(U , Ut);
	transpose(Vt , V);
	Result = V*W_inv*Ut*sceneMatrix;
	
	for (int i = 0; i < M; ++i) {
      Map_matrix.at<double>(i/3 , i%3) = Result.at<double>(i , 0);  //the column of v that corresponds to the smallest singular value,
																								 //which is the solution of the equations
	}
	Map_matrix.at<double>(2 , 2) = 1;
	//Denormalise
	Map_Matrix_return = ScenePtsTransformMat_Opt.inv(DECOMP_SVD)*Map_matrix*EyePtsTransformMat_Opt;
	
	return true;
}

bool sliceMapElement::mappingHomographyConstructionSelfDefined(const std::vector<Point> &calibratedCalPoints){
		Mat EyePtsTransformMat_Opt = Mat::zeros(3 , 3 , CV_64F);
		Mat ScenePtsTransformMat_Opt = Mat::zeros(3 , 3 , CV_64F);
		bool mappingDone = false;		
		Point2d scene_center, eye_center, *eye_nor, *scene_nor;
		double dis_scale_scene, dis_scale_eye;  			
		std::vector<Point> scenePoints_Set_in;
		std::vector<Point> eyePoints_Set_in;		
		

		//Normalize		
		scenePoints_Set_in.push_back(calibratedCalPoints[leftAnchorPt.second]);
		eyePoints_Set_in.push_back(leftAnchorPt.first);

		scenePoints_Set_in.push_back(calibratedCalPoints[upAnchorPt.second]);
		eyePoints_Set_in.push_back(upAnchorPt.first);

		scenePoints_Set_in.push_back(calibratedCalPoints[rightAnchorPt.second]);
		eyePoints_Set_in.push_back(rightAnchorPt.first);

		scenePoints_Set_in.push_back(calibratedCalPoints[downAnchorPt.second]);
		eyePoints_Set_in.push_back(downAnchorPt.first);
	
		scene_nor = normalize_edge_pointSet(dis_scale_scene, scene_center, 4 , scenePoints_Set_in);
		eye_nor = normalize_edge_pointSet(dis_scale_eye, eye_center, 4 , eyePoints_Set_in);


		//Forming transform matrix
		EyePtsTransformMat_Opt.at<double>(0 , 0) = dis_scale_eye;
		EyePtsTransformMat_Opt.at<double>(1 , 1) = dis_scale_eye;
		EyePtsTransformMat_Opt.at<double>(0 , 2) = -dis_scale_eye*eye_center.x;
		EyePtsTransformMat_Opt.at<double>(1 , 2) = -dis_scale_eye*eye_center.y;
		EyePtsTransformMat_Opt.at<double>(2 , 2) = 1;		

		ScenePtsTransformMat_Opt.at<double>(0 , 0) = dis_scale_scene;
		ScenePtsTransformMat_Opt.at<double>(1 , 1) = dis_scale_scene;
		ScenePtsTransformMat_Opt.at<double>(0 , 2) = -dis_scale_scene*scene_center.x;
		ScenePtsTransformMat_Opt.at<double>(1 , 2) = -dis_scale_scene*scene_center.y;
		ScenePtsTransformMat_Opt.at<double>(2 , 2) = 1;		


		//Solve Homography Mapping
		mappingDone = SolveHomography(scene_nor , eye_nor
															 , 4
															 , EyePtsTransformMat_Opt
															 , ScenePtsTransformMat_Opt
															 , HomoTransMatrix);		

		char test_file[MAX_WORD_LEN];
		sprintf(test_file , "HomoClass_%d.txt" , testOut);
		ofstream out_Homo(test_file);
		out_Homo<<"scenePoints_Set_in = "<<endl;
		for(int i=0;i<scenePoints_Set_in.size();++i){
			out_Homo<<scenePoints_Set_in[i]<<endl;
		}
		out_Homo<<"eyePoints_Set_in = "<<endl;
		for(int i=0;i<eyePoints_Set_in.size();++i){
			out_Homo<<eyePoints_Set_in[i]<<endl;
		}
		out_Homo<<"scene_nor:"<<endl;
		for(int i=0;i<4;++i){
			out_Homo<<scene_nor[i]<<endl;			
		}
		out_Homo<<"eye_nor:"<<endl;
		for(int i=0;i<4;++i){
			out_Homo<<eye_nor[i]<<endl;
		}
		out_Homo<<"EyePtsTransformMat_Opt = "<<endl<<EyePtsTransformMat_Opt<<endl;
		out_Homo<<"ScenePtsTransformMat_Opt = "<<endl<<ScenePtsTransformMat_Opt<<endl;
		out_Homo<<"HomoTransMatrix = "<<endl<<HomoTransMatrix<<endl;


		++testOut;
		return mappingDone;
	}


bool sliceMapElement::mappingHomographyConstruction(const std::vector<Point> &calibratedCalPoints){				
		std::vector<Point2d> scenePoints_Set_in;
		std::vector<Point2d> eyePoints_Set_in;		
				
		scenePoints_Set_in.push_back(calibratedCalPoints[leftAnchorPt.second]);
		eyePoints_Set_in.push_back(leftAnchorPt.first);

		scenePoints_Set_in.push_back(calibratedCalPoints[upAnchorPt.second]);
		eyePoints_Set_in.push_back(upAnchorPt.first);

		scenePoints_Set_in.push_back(calibratedCalPoints[rightAnchorPt.second]);
		eyePoints_Set_in.push_back(rightAnchorPt.first);

		scenePoints_Set_in.push_back(calibratedCalPoints[downAnchorPt.second]);
		eyePoints_Set_in.push_back(downAnchorPt.first);

		HomoTransMatrix = findHomography( eyePoints_Set_in, scenePoints_Set_in, cv::RANSAC );

		char test_file[MAX_WORD_LEN];
		sprintf(test_file , "HomoClass_%d.txt" , testOut);
		ofstream out_Homo(test_file);
		out_Homo<<"scenePoints_Set_in = "<<endl;
		for(int i=0;i<scenePoints_Set_in.size();++i){
			out_Homo<<scenePoints_Set_in[i]<<endl;
		}
		out_Homo<<"eyePoints_Set_in = "<<endl;
		for(int i=0;i<eyePoints_Set_in.size();++i){
			out_Homo<<eyePoints_Set_in[i]<<endl;
		}
		out_Homo<<"HomoTransMatrix = "<<endl<<HomoTransMatrix<<endl;


		++testOut;
		return true;
}



