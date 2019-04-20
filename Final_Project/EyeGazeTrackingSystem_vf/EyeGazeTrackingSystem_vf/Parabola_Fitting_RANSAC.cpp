#include "Parabola_Fitting_RANSAC.h"


inline float DistanceCaculate(const Point &x1 , const Point &x2){
	return max(abs((x1.x - x2.x)) , abs((x1.y - x2.y)))+0.5*min(abs((x1.x - x2.x)) , abs((x1.y - x2.y)));
}

inline float DistanceCaculate(const Point2f &x1 , const Point2f &x2){
	return max(fabs((x1.x - x2.x)) , fabs((x1.y - x2.y)))+0.5*min(fabs((x1.x - x2.x)) , fabs((x1.y - x2.y)));
}


inline const float& DistanceCaculateEuclidean(const Point2f &x1 , const Point2f &x2){
	Point2f vectorLineX1X2(x2.x - x1.x , x2.y - x1.y);
	return sqrtf(vectorLineX1X2.x*vectorLineX1X2.x + vectorLineX1X2.y*vectorLineX1X2.y);
}

inline Point2f* normalize_edge_pointType1(float &dis_scale, Point2f &nor_center, int ep_num 
																	, const vector<Point> &feature_point)
{
	const float sqrt_2 = 1.414213;
	float sumx = 0, sumy = 0;
	float sumdis = 0;
	Point edge;
	Point2f original(0.f , 0.f);

	//#pragma omp parallel for 
	for (int i = 0; i < ep_num; ++i){
		edge = feature_point.at(i);
		sumx += edge.x;
		sumy += edge.y;
		//sumdis += sqrtf((float)(edge.x*edge.x + edge.y*edge.y));
		//sumdis+=DistanceCaculate(edge,  original);
	}

	//dis_scale = sqrt_2*ep_num/sumdis;
	nor_center.x = sumx*1.0/ep_num;
	nor_center.y = sumy*1.0/ep_num;
	Point2f *edge_point_nor = (Point2f*)malloc(sizeof(Point2f)*ep_num);
	for (int i = 0; i < ep_num; ++i){
		edge = feature_point.at(i);
		edge_point_nor[i].x = ((float)edge.x - nor_center.x)/**dis_scale*/;
		edge_point_nor[i].y = ((float)edge.y - nor_center.y)/**dis_scale*/;		
	}

	for (int i = 0; i < ep_num; ++i){
		//sumdis+=DistanceCaculate(edge_point_nor[i],  original);	
		sumdis+=sqrtf((float)(edge_point_nor[i].x*edge_point_nor[i].x + edge_point_nor[i].y*edge_point_nor[i].y));
	}
	dis_scale = sqrt_2*ep_num/sumdis;

	for (int i = 0; i < ep_num; ++i){		
		edge_point_nor[i].x *= dis_scale;
		edge_point_nor[i].y *= dis_scale;		
	}

	return edge_point_nor;
}

inline Point2f* normalize_edge_pointType2(float &dis_scale, Point2f &nor_center, int ep_num 
																	, const vector<Point> &feature_point)
{
	const float sqrt_2 = 1.414213;
	float sumx = 0, sumy = 0;
	float sumdis = 0;
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
	Point2f *edge_point_nor = (Point2f*)malloc(sizeof(Point2f)*ep_num);
	for (int i = 0; i < ep_num; ++i){
		edge = feature_point.at(i);
		edge_point_nor[i].x = ((float)edge.x - nor_center.x)*dis_scale;
		edge_point_nor[i].y = ((float)edge.y - nor_center.y)*dis_scale;		
	}
	return edge_point_nor;
}

inline void get_5_random_num(int max_num, int* rand_num){
	int rand_index = 0;
	int r;
	int i;
	bool is_new = 1;

	if (max_num == 4) {
		for (i = 0; i < 5; ++i) {
			rand_num[i] = i;
		}
		return;
	}

	while (rand_index < 5) {
		is_new = 1;
		r = (int)((rand()*1.0/RAND_MAX) * max_num);
		for (i = 0; i < rand_index; i++) {
			if (r == rand_num[i]) {
				is_new = 0;
				break;
			}
		}
		if (is_new) {
			rand_num[rand_index] = r;
			rand_index++;
		}
	}
}

inline void get_3_random_num(int max_num, int* rand_num){
	int rand_index = 0;
	int r;
	int i;
	bool is_new = 1;

	if (max_num == 2) {
		for (i = 0; i < 3; ++i) {
			rand_num[i] = i;
		}
		return;
	}

	while (rand_index < 3) {
		is_new = 1;
		r = (int)((rand()*1.0/RAND_MAX) * max_num);
		for (i = 0; i < rand_index; ++i) {
			if (r == rand_num[i]) {
				is_new = 0;
				break;
			}
		}
		if (is_new) {
			rand_num[rand_index] = r;
			++rand_index;
		}
	}
}

inline void get_4_random_num(int max_num, int* rand_num){
	int rand_index = 0;
	int r;
	int i;
	bool is_new = 1;

	if (max_num == 3) {
		for (i = 0; i < 4; ++i) {
			rand_num[i] = i;
		}
		return;
	}

	while (rand_index < 4) {
		is_new = 1;
		r = (int)((rand()*1.0/RAND_MAX) * max_num);
		for (i = 0; i < rand_index; ++i) {
			if (r == rand_num[i]) {
				is_new = 0;
				break;
			}
		}
		if (is_new) {
			rand_num[rand_index] = r;
			++rand_index;
		}
	}
}

inline void get_1_random_num(int max_num, int* rand_num){
	int rand_index = 0;
	int r;
	int i;
	bool is_new = 1;

	if (max_num == 0) {
		for (i = 0; i < 1; ++i) {
			rand_num[i] = i;
		}
		return;
	}

	while (rand_index < 1) {
		is_new = 1;
		r = (int)((rand()*1.0/RAND_MAX) * max_num);
		for (i = 0; i < rand_index; ++i) {
			if (r == rand_num[i]) {
				is_new = 0;
				break;
			}
		}
		if (is_new) {
			rand_num[rand_index] = r;
			++rand_index;
		}
	}
}

// solve_ellipse
// conic_param[6] is the parameters of a conic {a, b, c, d, e, f}; conic equation: ax^2 + bxy + cy^2 + dx + ey + f = 0;
// ellipse_param[5] is the parameters of an ellipse {ellipse_a, ellipse_b, cx, cy, theta}; a & b is the major or minor axis; 
// cx & cy is the ellipse center; theta is the ellipse orientation
inline bool solve_ellipse(float* conic_param, float* ellipse_param){
	float a = conic_param[0];
	float b = conic_param[1];
	float c = conic_param[2];
	float d = conic_param[3];
	float e = conic_param[4];
	float f = conic_param[5];

	//determine whether it is an ellipse or circle
	float det_rule = b*b - 4*a*c;
	if(det_rule>=0){
		memset(ellipse_param, 0, sizeof(double)*5);
		return false;
	}

	//get ellipse orientation
	float theta = atan2(b, a-c)/2;

	//get scaled major/minor axes
	float ct = cos(theta);
	float st = sin(theta);
	float ap = a*ct*ct + b*ct*st + c*st*st;
	float cp = a*st*st - b*ct*st + c*ct*ct;

	//get translations
	float cx = (2*c*d - b*e) / (b*b - 4*a*c);
	float cy = (2*a*e - b*d) / (b*b - 4*a*c);

	//get scale factor
	//double val = a*cx*cx + b*cx*cy + c*cy*cy;//error? val = a*cx*cx + b*cx*cy + c*cy*cy + d*cx + e*cy
	//double scale_inv = val - f;//error? scal_inv = -val-f;
	float val = a*cx*cx + b*cx*cy + c*cy*cy + d*cx + e*cy;
	float scale_inv = -val - f;


	if (scale_inv/ap <= 0 || scale_inv/cp <= 0) {
	//printf("Error! ellipse parameters are imaginary a=sqrt(%lf), b=sqrt(%lf)\n", scale_inv/ap, scale_inv/cp);
		memset(ellipse_param, 0, sizeof(float)*5);
		return false;
	}

	ellipse_param[0] = sqrtf(scale_inv / ap);
	ellipse_param[1] = sqrtf(scale_inv / cp);
	ellipse_param[2] = cx;
	ellipse_param[3] = cy;
	ellipse_param[4] = theta;	
	
	return true;
}

inline void denormalize_ellipse_param(float* par, float* normailized_par, float dis_scale
														, Point2f nor_center)
{
    par[0] = normailized_par[0] / dis_scale;	//major or minor axis
    par[1] = normailized_par[1] / dis_scale;
    par[2] = normailized_par[2] / dis_scale + nor_center.x;	//ellipse center
    par[3] = normailized_par[3] / dis_scale + nor_center.y;
	par[4] = normailized_par[4];
}




inline bool DetermineOutsudeConics(const double * const conic_par , const Point2d &pt){
	return (conic_par[0]*pt.x*pt.x + 
			   conic_par[1]*pt.x*pt.y +
			   conic_par[2]*pt.y*pt.y + 
			   conic_par[3]*pt.x +
			   conic_par[4]*pt.y +
			   conic_par[5]>0)?true:false;
}

inline float DetermineOutsudeConicsReFloat(const float * conic_par , const Point2f &pt){
	return  conic_par[0]*pt.x*pt.x + 
			   conic_par[1]*pt.x*pt.y +
			   conic_par[2]*pt.y*pt.y + 
			   conic_par[3]*pt.x +
			   conic_par[4]*pt.y +
			   conic_par[5];
}

inline bool DetermineOutsudeConicsReFloat(const Mat &mask , const Point2f &pt){
	if(mask.at<uchar>(pt.y , pt.x)==255)
		return true;
	else
		return false;
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

inline Point2f DerivativeEllipse(const float * const conic_par , const Point2f &pt){
	return Point2f(2*conic_par[0]*pt.x+conic_par[1]*pt.y+conic_par[3] 
						,2*conic_par[2]*pt.y+conic_par[1]*pt.x+conic_par[5]);
}
inline Point2f DerivativeParabola(const float * const conic_par , const Point2f &pt){
	return Point2f(2*conic_par[0]*pt.x+conic_par[1]
						,-1);
}

inline float ErrorEllipseEOF2(const float * conic_par , const Point2f &pt , const Point2f &nor_center){
	Point2f derivative_conic(2*conic_par[0]*pt.x+conic_par[1]*pt.y+conic_par[3] ,
										2*conic_par[2]*pt.y+conic_par[1]*pt.x+conic_par[5]);	
	float magnitude = DistanceCaculate(derivative_conic,  nor_center);	
	//float magnitude = DistanceCaculateEuclidean(derivative_conic,  nor_center);

	return (DetermineOutsudeConicsReFloat(conic_par , pt)
			  /magnitude);
}

inline float ErrorEllipseEOF10(const float * conic_par , const Point2f &pt , const Point2f &ellip_center){	
	Point2f r_vector(ellip_center.x - pt.x , ellip_center.y - pt.y);
	//float magnitude = DistanceCaculate(r_vector,  Point2f(0.0 , 0.0));	
	float theta = atan2(r_vector.y , r_vector.x);

	cout<<"theta = "<<theta*180.0/M_PI;

	Point2f derivative_conic(2*conic_par[0]*pt.x+conic_par[1]*pt.y+conic_par[3] ,
										2*conic_par[2]*pt.y+conic_par[1]*pt.x+conic_par[5]);
	return DetermineOutsudeConicsReFloat(conic_par , pt)/(cos(theta)*derivative_conic.x+sin(theta)*derivative_conic.y);
}

inline float ErrorParabolaEOF1(const float *const conic_par , const Point2f &pt){
	float dis_error = conic_par[0]*pt.x*pt.x + 
	         				conic_par[1]*pt.x+
	         				conic_par[2]-(pt.y);
	return dis_error;
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

inline float ErrorCubicEOF1(const float *const conic_par , const Point2f &pt){
	float dis_error = conic_par[0]*pt.x*pt.x*pt.x + 
	         				conic_par[1]*pt.x*pt.x+
	         				conic_par[2]*pt.x + conic_par[3] - (pt.y);
	return dis_error;
}

inline void ConicDenormalize(const float *const conic_par , const float &dis_scale , const Point2f &nor_center 
											,float * conic_par_denor){

	float conic_par_denor_test[6] = {0.f};
	float matInit[3][3] = {
		{dis_scale , 0 , -dis_scale*nor_center.x},
		{0 , dis_scale , -dis_scale*nor_center.y},
		{0 , 0 , 1}
	};
	Mat TransformMatrix(3 , 3 , CV_32FC1 , matInit);
	Mat TransformMatrixTranspose;
	transpose(TransformMatrix, TransformMatrixTranspose);

	float A = conic_par[0];
	float B = conic_par[1];
	float C = conic_par[2];
	float D = conic_par[3];
	float E = conic_par[4];
	float F = conic_par[5];

	float matABCDEF[3][3] = {
		{A , B/2 , D/2},
		{B/2 , C , E/2},
		{D/2 ,  E/2 , F}
	};


	Mat ABCDEForiginal(3 , 3 , CV_32FC1 , matABCDEF);	
	Mat ABCDEFresult = TransformMatrixTranspose*ABCDEForiginal*TransformMatrix;	
	
	conic_par_denor[0] = ABCDEFresult.at<float>(0 , 0);
	conic_par_denor[1] = ABCDEFresult.at<float>(0 , 1)*2;
	conic_par_denor[2] = ABCDEFresult.at<float>(1 , 1);
	conic_par_denor[3] = ABCDEFresult.at<float>(2 , 0)*2;
	conic_par_denor[4] = ABCDEFresult.at<float>(2 , 1)*2;
	conic_par_denor[5] = ABCDEFresult.at<float>(2 , 2);
}

inline void ConicDenormalizeParabola(const float *const conic_par , const float &dis_scale , const Point2f &nor_center 
											,float * conic_par_denor){
	
	float matInit[3][3] = {
		{dis_scale , 0 , -dis_scale*nor_center.x},
		{0 , dis_scale , -dis_scale*nor_center.y},
		{0 , 0 , 1}
	};
	Mat TransformMatrix(3 , 3 , CV_32FC1 , matInit);
	Mat TransformMatrixTranspose;
	transpose(TransformMatrix, TransformMatrixTranspose);

	float A = conic_par[0];
	float B = conic_par[1];
	float C = conic_par[2];
	

	float matABCDEF[3][3] = {
		{A , 0 , B/2},
		{0 , 0 , -1/2},
		{B/2 , -1/2 , C}
	};


	Mat ABCDEForiginal(3 , 3 , CV_32FC1 , matABCDEF);	
	Mat ABCDEFresult = TransformMatrixTranspose*ABCDEForiginal*TransformMatrix;	
	
	conic_par_denor[0] = ABCDEFresult.at<float>(0 , 0);
	conic_par_denor[1] = ABCDEFresult.at<float>(0 , 2)*2;
	conic_par_denor[2] = ABCDEFresult.at<float>(2 , 2);	
}



inline bool EarlySampleRejection(const float * const conic_par , const Point2f* const edge_point_nor 
												, const int * const rand_index , const Mat &GradImageX , const Mat &GradImageY
												, const vector<Point> &feature_point){	
	for(int i=0;i<5;++i){
		Point2f test_pt(edge_point_nor[rand_index[i]]);	
		Point2f test_pt_denormalise(feature_point[rand_index[i]]);	
		Point2f grad_ellipse = DerivativeEllipse(conic_par, test_pt);
		Point2f grad_image(GradImageX.at<float>(test_pt_denormalise.y , test_pt_denormalise.x) 
									, GradImageY.at<float>(test_pt_denormalise.y , test_pt_denormalise.x));		
		if((grad_ellipse.x*grad_image.x + grad_ellipse.y*grad_image.y)<=0)
			return false;
	}		
	return true;
}

inline void GradientImageCreation(const Mat &Src , Mat &GradImageX , Mat &GradImageY){
	int size_gaussian = 27*(Src.rows/FRAMEH);
	if(size_gaussian%2==0)
		++size_gaussian;

	Mat Src_Gaussian;
	//Mat Grad_x, Grad_y;
	//Mat Abs_Grad_x, Abs_Grad_y;
	//Mat GradientImg;
	int scale = 1;
	int delta = 0;
	int ddepth = /*CV_16S*/CV_32FC1;
	//double time_a = omp_get_wtime();
	GaussianBlur( Src, Src_Gaussian, Size(size_gaussian,size_gaussian) , 0);	
	//double time_b = omp_get_wtime();
	//printf("Gaussian takes time : %lf\n" , time_b - time_a);

	//time_a = omp_get_wtime();
	Scharr( Src_Gaussian, GradImageX, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	//cout<<"Grad_x = "<<Grad_x<<endl;getchar();
	//convertScaleAbs( GradImageX, Abs_Grad_x );
	//time_b = omp_get_wtime();
	//printf("Scharrx takes time : %lf\n" , time_b - time_a);

	//time_a = omp_get_wtime();
	Scharr( Src_Gaussian, GradImageY, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	//convertScaleAbs( GradImageY, Abs_Grad_y );
	//time_b = omp_get_wtime();
	//printf("Scharry takes time : %lf\n" , time_b - time_a);

	//time_a = omp_get_wtime();
	//addWeighted( Abs_Grad_x, 0.5, Abs_Grad_y, 0.5, 0, GradientImg);
	//time_b = omp_get_wtime();
	//printf("addWeighted takes time : %lf\n" , time_b - time_a);	
	//imshow("GradImage" , GradientImg);
}

inline float ImageAwareSupport(const float *const conic_par , const Mat &GradImageX , const Mat &GradImageY
												, const int * const inliers_index , const int &ninliers , const Point2f* const edge_point_nor 
												, const vector<Point> &feature_point){
	float sum = 0;
	for(int i=0;i<ninliers;++i){
		Point2f test_pt(edge_point_nor[inliers_index[i]]);
		Point2f test_pt_denormalise(feature_point[inliers_index[i]]);
		Point2f grad_ellipse = DerivativeEllipse(conic_par, test_pt);
		Point2f grad_image(GradImageX.at<float>(test_pt_denormalise.y , test_pt_denormalise.x) 
									, GradImageY.at<float>(test_pt_denormalise.y , test_pt_denormalise.x));		
		float grad_ellipse_length = sqrtf(grad_ellipse.x*grad_ellipse.x + grad_ellipse.y*grad_ellipse.y);

		//normalise
		grad_ellipse.x/=grad_ellipse_length;
		grad_ellipse.y/=grad_ellipse_length;

		sum+=grad_ellipse.x*grad_image.x + grad_ellipse.y*grad_image.y;
	}
	return sum;
}

inline float ImageAwareSupportParabola(const float *const conic_par , const Mat &GradImageX , const Mat &GradImageY
												, const vector<Point> &inlier_parabola, const vector<Point> &feature_point){
	float sum = 0;
	for(int i=0;i<inlier_parabola.size();++i){
		Point2f test_pt(inlier_parabola[i]);		
		Point2f grad_parabola = DerivativeParabola(conic_par, test_pt);
		Point2f grad_image(GradImageX.at<float>(test_pt.y , test_pt.x) 
									, GradImageY.at<float>(test_pt.y , test_pt.x));		
		float grad_parabola_length = sqrtf(grad_parabola.x*grad_parabola.x + grad_parabola.y*grad_parabola.y);
		float grad_image_length = sqrtf(grad_image.x*grad_image.x + grad_image.y*grad_image.y);

		//normalise		
		grad_parabola.x/=grad_parabola_length;
		grad_parabola.y/=grad_parabola_length;		
		grad_image.x/=grad_image_length;
		grad_image.y/=grad_image_length;	

		sum+=grad_parabola.x*grad_image.x + grad_parabola.y*grad_image.y;
	}	

	return sum;
}

inline float GeometrySupport(const float *const conic_par , const int &ninliers){
	float support = conic_par[2]*0.9+ ninliers*0.1;
	return support;
}

inline bool CheckISParabolas(const float *const conic_par){
	if(fabs(conic_par[1]*conic_par[1] - 4*conic_par[0]*conic_par[2])<0.0000000000001)
		return true;
	else
		return false;
}

inline void SolveCubicParameter(const vector<Point> &edge_point , const int *const rand_index
													, float *conic_par){
	 double x1, y1, x2, y2, x3, y3 , x4 , y4;
     x1 = (double)edge_point[rand_index[0]].x;
     y1 = (double)edge_point[rand_index[0]].y;
     x2 = (double)edge_point[rand_index[1]].x;
     y2 = (double)edge_point[rand_index[1]].y;
     x3 = (double)edge_point[rand_index[2]].x;
     y3 = (double)edge_point[rand_index[2]].y;
	 x4 = (double)edge_point[rand_index[3]].x;
	 y4 = (double)edge_point[rand_index[3]].y;


	 float matInit[4][4] = {
		{x1*x1*x1 , x1*x1  , x1 , 1},
		{x2*x2*x2 , x2*x2  , x2 , 1},
		{x3*x3*x3 , x3*x3  , x3 , 1},
		{x4*x4*x4 , x4*x4  , x4 , 1}
	};

	 float matInitY[4][1] = {
		 {y1},
		 {y2},
		 {y3},
		 {y4}
	 };

	Mat AMatrix(4 , 4 , CV_32FC1 , matInit);
	Mat YMatrix(4 , 1 , CV_32FC1 , matInitY);
	Mat Result = AMatrix.inv()*YMatrix;

	//cout<<"Result = "<<Result<<endl;

	conic_par[0]  = Result.at<float>(0 , 0);
	conic_par[1]  = Result.at<float>(1 , 0);
	conic_par[2]  = Result.at<float>(2 , 0);
	conic_par[3]  = Result.at<float>(3 , 0);
	//Mat TransformMatrixTranspose;
	//transpose(TransformMatrix, TransformMatrixTranspose);


	//Type2
	//double denom = (x1 - x2)*(x1 - x3)*(x2 - x3);
	//double A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom;
	//double B = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / denom;
	//double C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom;

	//conic_par[0]  = A;
	//conic_par[1]  = B;
	//conic_par[2]  = C;




	//Type1
     //conic_par[0] = ((y1-y2)/(x1-x2)-(y1-y3)/(x1-x3)) / (x2-x3);
     //conic_par[1] = (y1-y2)/(x1-x2)-conic_par[0]*(x1+x2);
     //conic_par[2] = y1-conic_par[0]*x1*x1-conic_par[1]*x1;
}
inline void SolveParabolaParameter(const vector<Point> &edge_point , const int *const rand_index
													, float *conic_par){
	 double x1, y1, x2, y2, x3, y3;
     x1 = (double)edge_point[rand_index[0]].x;
     y1 = (double)edge_point[rand_index[0]].y;
     x2 = (double)edge_point[rand_index[1]].x;
     y2 = (double)edge_point[rand_index[1]].y;
     x3 = (double)edge_point[rand_index[2]].x;
     y3 = (double)edge_point[rand_index[2]].y;


	 float matInit[3][3] = {
		{x1*x1 , x1 , 1},
		{x2*x2 , x2 , 1},
		{x3*x3 , x3 , 1}
	};

	 float matInitY[3][1] = {
		 {y1},
		 {y2},
		 {y3}
	 };

	Mat AMatrix(3 , 3 , CV_32FC1 , matInit);
	Mat YMatrix(3 , 1 , CV_32FC1 , matInitY);
	Mat Result = AMatrix.inv()*YMatrix;

	//cout<<"Result = "<<Result<<endl;

	conic_par[0]  = Result.at<float>(0 , 0);
	conic_par[1]  = Result.at<float>(1 , 0);
	conic_par[2]  = Result.at<float>(2 , 0);
	//Mat TransformMatrixTranspose;
	//transpose(TransformMatrix, TransformMatrixTranspose);


	//Type2
	//double denom = (x1 - x2)*(x1 - x3)*(x2 - x3);
	//double A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom;
	//double B = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / denom;
	//double C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom;

	//conic_par[0]  = A;
	//conic_par[1]  = B;
	//conic_par[2]  = C;




	//Type1
     //conic_par[0] = ((y1-y2)/(x1-x2)-(y1-y3)/(x1-x3)) / (x2-x3);
     //conic_par[1] = (y1-y2)/(x1-x2)-conic_par[0]*(x1+x2);
     //conic_par[2] = y1-conic_par[0]*x1*x1-conic_par[1]*x1;
}

inline void SolveParabolaParameterWEyeCorner(const vector<Point> &edge_point , const int *const rand_index
													, float *conic_par , const Point &leftEyeCorner , const Point &rightEyeCorner){
	 double x1, y1, x2, y2, x3, y3;
     x1 = (double)leftEyeCorner.x;
     y1 = (double)leftEyeCorner.y;
     x2 = (double)rightEyeCorner.x;
     y2 = (double)rightEyeCorner.y;
     x3 = (double)edge_point[rand_index[0]].x;
     y3 = (double)edge_point[rand_index[0]].y;


	 float matInit[3][3] = {
		{x1*x1 , x1 , 1},
		{x2*x2 , x2 , 1},
		{x3*x3 , x3 , 1}
	};

	 float matInitY[3][1] = {
		 {y1},
		 {y2},
		 {y3}
	 };

	Mat AMatrix(3 , 3 , CV_32FC1 , matInit);
	Mat YMatrix(3 , 1 , CV_32FC1 , matInitY);
	Mat Result = AMatrix.inv()*YMatrix;

	//cout<<"Result = "<<Result<<endl;

	conic_par[0]  = Result.at<float>(0 , 0);
	conic_par[1]  = Result.at<float>(1 , 0);
	conic_par[2]  = Result.at<float>(2 , 0);
	//Mat TransformMatrixTranspose;
	//transpose(TransformMatrix, TransformMatrixTranspose);


	//Type2
	//double denom = (x1 - x2)*(x1 - x3)*(x2 - x3);
	//double A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom;
	//double B = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / denom;
	//double C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom;

	//conic_par[0]  = A;
	//conic_par[1]  = B;
	//conic_par[2]  = C;




	//Type1
     //conic_par[0] = ((y1-y2)/(x1-x2)-(y1-y3)/(x1-x3)) / (x2-x3);
     //conic_par[1] = (y1-y2)/(x1-x2)-conic_par[0]*(x1+x2);
     //conic_par[2] = y1-conic_par[0]*x1*x1-conic_par[1]*x1;
}


void Parabola_Fitting_RANSACUp(const Mat &Src , const vector<Point> &feature_point ,float* &parabola_param
											, vector<Point> &inlier_parabolaReturn , vector<Point> &inlier_parabolaSoftReturn 
											, vector<Point> &outlier_parabola , Point &vertexParabolaUp , float &dis_thresholdSoft)
{
	
	int ep_num = feature_point.size();   //ep stands for edge point	
	Point2f nor_center;
	float dis_scale;	
	parabola_param = new float [3]();
	vector<Point> inlier_parabola;
	vector<Point> inlier_parabolaSoft;

	
	int parabola_point_num = 3;	//number of point that needed to fit an ellipse
	if (ep_num < parabola_point_num){
		printf("Error! %d points are not enough to fit parabola\n", ep_num);						
		return;
	}

	//Normalization
	Point2f *edge_point_nor = normalize_edge_pointType2(dis_scale, nor_center, ep_num , feature_point);
	
	

	//Ransac	
	long max_inliers = -LONG_MAX;
	float max_support = -FLT_MAX;
	float max_4c = -FLT_MAX;
	int sample_num = /*100;*/ep_num*3;	//number of sample
	int ransac_count = 0;


	float dis_threshold = 3000*dis_scale;	
	dis_thresholdSoft = /*12000*/8000*dis_scale;		
	float dis_error;
	float parabola_4c_threshold = 8000*FRAMEH/Src.rows*FRAMEW/Src.cols;
  	
	int rand_index[3];
	float conic_par[3] = {0.f};
	float conic_par_denor[3] = {0.f};
	float ratio;
	
	float inliners_avg_rate = 0;
	int inliers_countfraem = 0;

	/*Mat GradImageX , GradImageY;
	GradientImageCreation(Src , GradImageX ,GradImageY);*/
	//-----------------------------------------------------RANSAC begins---------------------------------------------------------//	
	//cout<<"============================="<<endl;
	while (sample_num > ransac_count){	
		inlier_parabola.clear();		
		inlier_parabolaSoft.clear();		
			
		++ransac_count;	
		

		//---------------------------Randomly Get Five Points---------------------------//
		get_3_random_num((ep_num-1), rand_index);


		//---------------------------Solve Parabola Parameters---------------------------//
		SolveParabolaParameter(feature_point , rand_index,conic_par_denor);
		Point vertexCenter(-conic_par_denor[1]/(2*conic_par_denor[0]) , conic_par_denor[2] - 
										conic_par_denor[1]*conic_par_denor[1]/(4*conic_par_denor[0]));

		//---------------------------Geometry Filter---------------------------//									
		//if(fabs(1.f/conic_par_denor[0])<parabola_4c_threshold || fabs(1.f/conic_par_denor[0])>parabola_4c_threshold*1.5)continue;		
		if(conic_par_denor[0]>=0)continue;
		if(vertexCenter.y>FRAMEH || vertexCenter.y<0 )continue;	
		if(vertexCenter.x>FRAMEW || vertexCenter.x<0 )continue;	


		////if(vertexCenter.x>FRAMEW || vertexCenter.x<0 )continue;		
		//if(fabs(float(vertexCenter.x - vertexParabolaDown.x))>200)continue;
		////if(fabs(float(vertexCenter.x - ellipse_par[2]))>100)continue;
		//if(!(eyelidPosibbleRangeMin<vertexCenter.y && vertexCenter.y<eyelidPosibbleRangeMax))continue;
		////if((vertexCenter.y - ellipseCenter.y)<100)continue;
		

		

		//---------------------------EOF Estimation---------------------------//	
		for (int i = 0; i < ep_num; ++i){			
			dis_error = //ErrorParabolaEOF1(conic_par_denor , feature_point[i]);			
							ErrorParabolaEOF2(conic_par_denor , feature_point[i]);
			if (fabs(dis_error) < dis_threshold){
				inlier_parabola.push_back(feature_point[i]);
				inlier_parabolaSoft.push_back(feature_point[i]);
			}else if(fabs(dis_error) < dis_thresholdSoft){
				inlier_parabolaSoft.push_back(feature_point[i]);
			}						
		}	

		//int bonus = 0;
		//if(vertexCenter.y<FRAMEH && vertexCenter.y>0){
		//	bonus = vertexCenter.y*1000000;
		//}else{
		//	bonus = 0;
		//}
		//long bonus = 0;/*fabs(1.f/conic_par_denor[0]);*/

		/*float supportScore = ImageAwareSupportParabola(conic_par_denor , GradImageX , GradImageY 
																					, inlier_parabola , feature_point);*/
		if((long)inlier_parabola.size()>max_inliers){
		//if(supportScore>max_support){
			inlier_parabolaReturn.clear();
			inlier_parabolaSoftReturn.clear();

			inlier_parabolaReturn = inlier_parabola;
			inlier_parabolaSoftReturn = inlier_parabolaSoft;
			for (int i = 0; i < 3; ++i) {				
				parabola_param[i] = conic_par_denor[i];
			}
			max_inliers = (long)inlier_parabolaReturn.size();		
			//max_support = supportScore;
			vertexParabolaUp = vertexCenter;			
		}
	}//end while
	

	for(int i=0;i<feature_point.size();++i){
		bool isOutlier = true;
		for(int j=0;j<inlier_parabolaSoftReturn.size();++j){
			if(feature_point[i]==inlier_parabolaSoftReturn[j])
				isOutlier = false;
		}
		if(isOutlier)
			outlier_parabola.push_back(feature_point[i]);	
	}

	free(edge_point_nor);
	edge_point_nor = NULL;	
	return ;

}//RANSAC PARABOLA UP






void Parabola_Fitting_RANSACDown(const Mat &Src , const vector<Point> &feature_point ,float* &parabola_param
											,  vector<Point> &inlier_parabolaReturn , vector<Point> &inlier_parabolaSoftReturn 
											,  vector<Point> &outlier_parabola , Point &vertexParabolaDown , float &dis_thresholdSoft)
{
	
	int ep_num = feature_point.size();   //ep stands for edge point	
	Point2f nor_center;
	float dis_scale;	
	parabola_param = new float [3]();
	vector<Point> inlier_parabola;
	vector<Point> inlier_parabolaSoft;	

	
	int parabola_point_num = 3;	//number of point that needed to fit an ellipse
	if (ep_num < parabola_point_num){
		printf("Error! %d points are not enough to fit parabola\n", ep_num);						
		return;
	}

	//Normalization
	Point2f *edge_point_nor = normalize_edge_pointType2(dis_scale, nor_center, ep_num , feature_point);
	
	

	//Ransac	
	int max_inliers = -INT_MAX;
	float max_support = -FLT_MAX;
	float max_4c = -FLT_MAX;
	int sample_num = /*100;*/ep_num*3;	//number of sample
	int ransac_count = 0;


	float dis_threshold = 3000*dis_scale;	
	dis_thresholdSoft = /*15000*/9000*dis_scale;	
	float dis_error;
	//float parabola_4c_threshold = 800*FRAMEH/Src.rows*FRAMEW/Src.cols;
	//float parabola_4c_threshold = norm(rightEyeCorner - leftEyeCorner)*FRAMEH/Src.rows*FRAMEW/Src.cols;
  

	int rand_index[3];
	float conic_par[3] = {0.f};
	float conic_par_denor[3] = {0.f};
	float ratio;
	
	float inliners_avg_rate = 0;
	int inliers_countfraem = 0;

	/*Mat GradImageX , GradImageY;
	GradientImageCreation(Src , GradImageX ,GradImageY);*/
	//-----------------------------------------------------RANSAC begins---------------------------------------------------------//		
	while (sample_num > ransac_count){	
		inlier_parabola.clear();		
		inlier_parabolaSoft.clear();		
			
		++ransac_count;	
		

		//---------------------------Randomly Get Five Points---------------------------//
		get_3_random_num((ep_num-1), rand_index);


		//---------------------------Solve Parabola Parameters---------------------------//
		SolveParabolaParameter(feature_point , rand_index,conic_par_denor);
		Point vertexCenter(-conic_par_denor[1]/(2*conic_par_denor[0]) , conic_par_denor[2] - 
										conic_par_denor[1]*conic_par_denor[1]/(4*conic_par_denor[0]));

		//---------------------------Geometry Filter---------------------------//									
		//if(fabs(1.f/conic_par_denor[0])<parabola_4c_threshold || fabs(1.f/conic_par_denor[0])>parabola_4c_threshold*1.5)continue;
		if (conic_par_denor[0]<=0)continue;
		if(vertexCenter.x<0 || vertexCenter.x>FRAMEW)continue;
		if(vertexCenter.y<0 || vertexCenter.y>FRAMEH)continue;
		


		//---------------------------EOF Estimation---------------------------//	
		for (int i = 0; i < ep_num; ++i){			
			dis_error = //ErrorParabolaEOF1(conic_par_denor , feature_point[i]);	
							ErrorParabolaEOF2(conic_par_denor , feature_point[i]);

			if (fabs(dis_error) < dis_threshold){
				inlier_parabola.push_back(feature_point[i]);
				inlier_parabolaSoft.push_back(feature_point[i]);
			}else if(fabs(dis_error) < dis_thresholdSoft){
				inlier_parabolaSoft.push_back(feature_point[i]);
			}

		}	
					

		/*float supportScore = ImageAwareSupportParabola(conic_par_denor , GradImageX , GradImageY 
																					, inlier_parabola , feature_point);*/

		if ((int)inlier_parabola.size() > max_inliers){			
		//if(supportScore>max_support){
			inlier_parabolaReturn.clear();
			inlier_parabolaSoftReturn.clear();

			inlier_parabolaReturn = inlier_parabola;
			inlier_parabolaSoftReturn = inlier_parabolaSoft;
			for (int i = 0; i < 3; ++i) {				
				parabola_param[i] = conic_par_denor[i];
			}
			max_inliers = (int)inlier_parabolaReturn.size();		
			//max_support = supportScore;		
			vertexParabolaDown = vertexCenter;
		}
	}//end while
	

	for(int i=0;i<feature_point.size();++i){
		bool isOutlier = true;
		for(int j=0;j<inlier_parabolaSoftReturn.size();++j){
			if(feature_point[i]==inlier_parabolaSoftReturn[j])
				isOutlier = false;
		}
		if(isOutlier)
			outlier_parabola.push_back(feature_point[i]);	
	}


	free(edge_point_nor);
	edge_point_nor = NULL;	
	return ;

}//RANSAC PARABOLA Down


