#ifndef TBB_PARALLAL_SPEEDUP_H
#define TBB_PARALLAL_SPEEDUP_H
#pragma once

#include<opencv2\opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <map>
using namespace cv;

class Parallel_process_gau : public cv::ParallelLoopBody
    {

    private:
        Mat img;
        Mat& retVal;
        const int size;
        int diff;
    public:
        Parallel_process_gau(cv::Mat inputImgage, cv::Mat& outImage, 
                        const int &sizeVal, int diffVal)
                    : img(inputImgage), retVal(outImage), 
                      size(sizeVal), diff(diffVal){}

        virtual void operator()(const cv::Range& range) const
        {

			//#pragma omp parallel for		
            for(int i = range.start; i < range.end; ++i)
            {
                /* divide image in 'diff' number 
                of parts and process simultaneously */
                cv::Mat in(img, cv::Rect(0, (img.rows/diff)*i, 
                           img.cols, img.rows/diff));				
                cv::Mat out(retVal, cv::Rect(0, (retVal.rows/diff)*i, 
                                    retVal.cols, retVal.rows/diff));

				GaussianBlur( in, out, Size(size,size) , 0);		
            }
        }
};

class Parallel_process_er : public cv::ParallelLoopBody
    {

    private:
        Mat img;
        Mat& retVal;
		Mat element;
        const int size;
		int struc_elem;
        int diff;
    public:
        Parallel_process_er(cv::Mat inputImgage, cv::Mat& outImage, 
                         const int &sizeVal, int struc_ele, int diffVal)
                    : img(inputImgage), retVal(outImage), 
                      size(sizeVal) , struc_elem(struc_ele), diff(diffVal){}

        virtual void operator()(const cv::Range& range) const
        {

			//#pragma omp parallel for
			Mat element = getStructuringElement( struc_elem, Size( 2*size + 1, 2*size+1 ), Point( size, size ) );
            for(int i = range.start; i < range.end; ++i)
            {
                /* divide image in 'diff' number 
                of parts and process simultaneously */

                cv::Mat in(img, cv::Rect(0, (img.rows/diff)*i, 
                           img.cols, img.rows/diff));			
                cv::Mat out(retVal, cv::Rect(0, (retVal.rows/diff)*i, 
                                    retVal.cols, retVal.rows/diff));
				//Morphology_Operations(in, tmp , MORPH_OPEN, openint_size,  MORPH_RECT);						
				/// Apply the specified morphology operation
				morphologyEx( in, out, MORPH_ERODE, element ,Point(-1,-1), 1,BORDER_DEFAULT);  				
            }
        }
};

class Parallel_process_di : public cv::ParallelLoopBody
    {

    private:
        Mat img;
        Mat& retVal;
		Mat element;
        const int size;
		int struc_elem;
        int diff;
    public:
        Parallel_process_di(cv::Mat inputImgage, cv::Mat& outImage, 
                         const int &sizeVal, int struc_ele, int diffVal)
                    : img(inputImgage), retVal(outImage), 
					size(sizeVal), struc_elem(struc_ele), diff(diffVal){}

        virtual void operator()(const cv::Range& range) const
        {

			//#pragma omp parallel for
			Mat element = getStructuringElement( struc_elem, Size( 2*size + 1, 2*size+1 ), Point( size, size ) );
            for(int i = range.start; i < range.end; ++i)
            {
                /* divide image in 'diff' number 
                of parts and process simultaneously */

                cv::Mat in(img, cv::Rect(0, (img.rows/diff)*i, 
                           img.cols, img.rows/diff));			
                cv::Mat out(retVal, cv::Rect(0, (retVal.rows/diff)*i, 
                                    retVal.cols, retVal.rows/diff));
				//Morphology_Operations(in, tmp , MORPH_OPEN, openint_size,  MORPH_RECT);						
				/// Apply the specified morphology operation				
				morphologyEx( in, out, MORPH_DILATE, element ,Point(-1,-1), 1,BORDER_DEFAULT);  
							
            }
        }
};

class Parallel_process_thre : public cv::ParallelLoopBody
    {

    private:
        Mat img;
        Mat& retVal;
		Mat element;        
        int diff;
    public:
        Parallel_process_thre(cv::Mat inputImgage, cv::Mat& outImage, 
                         int diffVal)
                    : img(inputImgage), retVal(outImage), 
                      diff(diffVal){}

        virtual void operator()(const cv::Range& range) const
        {

			//#pragma omp parallel for			
            for(int i = range.start; i < range.end; ++i)
            {
                /* divide image in 'diff' number 
                of parts and process simultaneously */
                cv::Mat in(img, cv::Rect(0, (img.rows/diff)*i, 
                           img.cols, img.rows/diff));			
                cv::Mat out(retVal, cv::Rect(0, (retVal.rows/diff)*i, 
                                    retVal.cols, retVal.rows/diff));
				threshold(in, out, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
							
            }
        }
};

class Parallel_process_threBinary : public cv::ParallelLoopBody
    {

    private:
        Mat img;
        Mat& retVal;
		const int thresholdValue;
        int diff;
    public:
        Parallel_process_threBinary(cv::Mat inputImgage, cv::Mat& outImage, 
                         const int &thresh_value, int diffVal)
                    : img(inputImgage), retVal(outImage), 
					thresholdValue(thresh_value), diff(diffVal){}

        virtual void operator()(const cv::Range& range) const
        {

			//#pragma omp parallel for			
            for(int i = range.start; i < range.end; ++i)
            {
                /* divide image in 'diff' number 
                of parts and process simultaneously */
                cv::Mat in(img, cv::Rect(0, (img.rows/diff)*i, 
                           img.cols, img.rows/diff));			
                cv::Mat out(retVal, cv::Rect(0, (retVal.rows/diff)*i, 
                                    retVal.cols, retVal.rows/diff));				
				threshold(in, out, thresholdValue, 255,THRESH_BINARY);							
            }
        }
};


class Parallel_process_scharrX: public cv::ParallelLoopBody
    {

    private:
        Mat img;
        Mat& retVal;       
        int diff;
    public:
        Parallel_process_scharrX(cv::Mat inputImgage, cv::Mat& outImage, 
                         int diffVal)
                    : img(inputImgage), retVal(outImage), 
                      diff(diffVal){}

        virtual void operator()(const cv::Range& range) const
        {

			//#pragma omp parallel for		
            for(int i = range.start; i < range.end; ++i)
            {
                /* divide image in 'diff' number 
                of parts and process simultaneously */

                cv::Mat in(img, cv::Rect(0, (img.rows/diff)*i, 
                           img.cols, img.rows/diff));			
                cv::Mat out(retVal, cv::Rect(0, (retVal.rows/diff)*i, 
                                    retVal.cols, retVal.rows/diff));
				Scharr( in, out, CV_16S, 1, 0, 3, 0, BORDER_DEFAULT );					
            }
        }
};


class Parallel_process_scharrY: public cv::ParallelLoopBody
    {

    private:
        Mat img;
        Mat& retVal;       
        int diff;
    public:
        Parallel_process_scharrY(cv::Mat inputImgage, cv::Mat& outImage, 
                         int diffVal)
                    : img(inputImgage), retVal(outImage), 
                      diff(diffVal){}

        virtual void operator()(const cv::Range& range) const
        {

			//#pragma omp parallel for		
            for(int i = range.start; i < range.end; ++i)
            {
                /* divide image in 'diff' number 
                of parts and process simultaneously */

                cv::Mat in(img, cv::Rect(0, (img.rows/diff)*i, 
                           img.cols, img.rows/diff));			
                cv::Mat out(retVal, cv::Rect(0, (retVal.rows/diff)*i, 
                                    retVal.cols, retVal.rows/diff));

				Scharr( in, out, CV_16S, 0, 1, 3, 0, BORDER_DEFAULT );				
            }
        }
};

class Parallel_process_convertScaleAbs : public cv::ParallelLoopBody
    {

    private:
        Mat img;
        Mat& retVal;
		Mat element;        
        int diff;
    public:
        Parallel_process_convertScaleAbs(cv::Mat inputImgage, cv::Mat& outImage, 
                         int diffVal)
                    : img(inputImgage), retVal(outImage), 
                      diff(diffVal){}

        virtual void operator()(const cv::Range& range) const
        {

			//#pragma omp parallel for		
            for(int i = range.start; i < range.end; ++i)
            {
                /* divide image in 'diff' number 
                of parts and process simultaneously */

                cv::Mat in(img, cv::Rect(0, (img.rows/diff)*i, 
                           img.cols, img.rows/diff));			
                cv::Mat out(retVal, cv::Rect(0, (retVal.rows/diff)*i, 
                                    retVal.cols, retVal.rows/diff));

				convertScaleAbs( in, out );										
            }
        }
};
class Parallel_process_normalize : public cv::ParallelLoopBody
    {

    private:
        Mat img;
        Mat& retVal;
		Mat element;        
        int diff;
    public:
        Parallel_process_normalize(cv::Mat inputImgage, cv::Mat& outImage, 
                         int diffVal)
                    : img(inputImgage), retVal(outImage), 
                      diff(diffVal){}

        virtual void operator()(const cv::Range& range) const
        {

			//#pragma omp parallel for		
            for(int i = range.start; i < range.end; ++i)
            {
                /* divide image in 'diff' number 
                of parts and process simultaneously */

                cv::Mat in(img, cv::Rect(0, (img.rows/diff)*i, 
                           img.cols, img.rows/diff));			
                cv::Mat out(retVal, cv::Rect(0, (retVal.rows/diff)*i, 
                                    retVal.cols, retVal.rows/diff));

				normalize(in, out, 0, 255, NORM_MINMAX, CV_8UC1);//L							
            }
        }
};




class Parallel_process_bitwiseand : public cv::ParallelLoopBody
    {

    private:
        Mat imgA;
		Mat imgB;
        Mat& retVal;
		Mat element;        
        int diff;
    public:
        Parallel_process_bitwiseand(cv::Mat inputImgageA, cv::Mat inputImgageB, cv::Mat& outImage, 
                         int diffVal)
                    : imgA(inputImgageA), imgB(inputImgageB), retVal(outImage), 
                      diff(diffVal){}

        virtual void operator()(const cv::Range& range) const
        {

			//#pragma omp parallel for		
            for(int i = range.start; i < range.end; ++i)
            {
                /* divide image in 'diff' number 
                of parts and process simultaneously */
                cv::Mat inA(imgA, cv::Rect(0, (imgA.rows/diff)*i, 
                           imgA.cols, imgA.rows/diff));
				cv::Mat inB(imgB, cv::Rect(0, (imgB.rows/diff)*i, 
                           imgB.cols, imgB.rows/diff));		
                cv::Mat out(retVal, cv::Rect(0, (retVal.rows/diff)*i, 
                                    retVal.cols, retVal.rows/diff));

				bitwise_and(inA , inB, out);	
							
            }
        }
};


class Parallel_process_displayIrisRegion : public cv::ParallelLoopBody
    {

    private:
        Mat img;		
        Mat& retVal;
		Mat element;        
        int diff;
    public:
        Parallel_process_displayIrisRegion(cv::Mat inputImgage, cv::Mat& outImage, 
                         int diffVal)
                    : img(inputImgage), retVal(outImage), 
                      diff(diffVal){}

        virtual void operator()(const cv::Range& range) const
        {

			//#pragma omp parallel for		
            for(int i = range.start; i < range.end; ++i)
            {
                /* divide image in 'diff' number 
                of parts and process simultaneously */
                cv::Mat inA(img, cv::Rect(0, (img.rows/diff)*i, 
                           img.cols, img.rows/diff));			
                cv::Mat out(retVal, cv::Rect(0, (retVal.rows/diff)*i, 
                                    retVal.cols, retVal.rows/diff));

				for(int j=0;i<inA.rows;++i){
					for(int k=0;j<inA.cols;++j){
						if(inA.at<uchar>(j , k)==255){
							out.at<Vec3b>(j , k) = Vec3b(255 , 255 , 255);
						}
					}
				}		
							
            }
        }
};

class Parallel_process_sub : public cv::ParallelLoopBody
{

private:
	Mat img;
	Mat& retVal;
	int diff;
public:
	Parallel_process_sub(cv::Mat inputImgage, cv::Mat& outImage, int diffVal)
		: img(inputImgage), retVal(outImage), diff(diffVal){}

	virtual void operator()(const cv::Range& range) const
	{

		//#pragma omp parallel for		
		for (int i = range.start; i < range.end; ++i)
		{
			/* divide image in 'diff' number
			of parts and process simultaneously */
			cv::Mat in(img, cv::Rect(0, (img.rows / diff)*i,
				img.cols, img.rows / diff));
			cv::Mat out(retVal, cv::Rect(0, (retVal.rows / diff)*i,
				retVal.cols, retVal.rows / diff));
			out = Scalar::all(255) - in;
		}
	}
};

class Parallel_process_hist : public cv::ParallelLoopBody
{

private:
	Mat img;
	int diff;
	int *hist;
	std::mutex &mtx;
	int  &total_hist_sum;

public:
	Parallel_process_hist(cv::Mat inputImgage, int *hist, std::mutex &mtx, int &total_hist_sum, int diffVal)
		: img(inputImgage), diff(diffVal), hist(hist), mtx(mtx), total_hist_sum(total_hist_sum){}

	virtual void operator()(const cv::Range& range) const
	{	
		for (int i = range.start; i < range.end; ++i)
		{
			/* divide image in 'diff' number
			of parts and process simultaneously */
			cv::Mat in(img, cv::Rect(0, (img.rows / diff)*i,
				img.cols, img.rows / diff));
			int *local_hist = new int[256]();
			double local_hist_sum = 0;

			for (int r = 0; r<in.rows; ++r) {
				for (int c = 0; c<in.cols; ++c) {
					int temp = in.at<uchar>(r, c);
					local_hist[temp]++;
				}
			}

			for (int j = 0; j < 256; ++j) {
				local_hist_sum += j*local_hist[j];
			}

			mtx.lock();
			for (int k = 0; k < 256; ++k) { hist[k] += local_hist[k]; }
			total_hist_sum += local_hist_sum;
			mtx.unlock();
		}
	}
};

class Parallel_process_cal_thresh : public cv::ParallelLoopBody
{

private:
	int diff;
	int *hist;
	std::mutex &mtx;
	int total_hist_sum;
	int *threshold_cand;
	double *var_cand;

public:
	Parallel_process_cal_thresh(int *hist, std::mutex &mtx, int total_hist_sum, int*threshold_cand, double *var_cand, int diffVal)
		: diff(diffVal), hist(hist), mtx(mtx), total_hist_sum(total_hist_sum), threshold_cand(threshold_cand), var_cand(var_cand) {}


	virtual void operator()(const cv::Range& range) const
	{
		for (int i = range.start; i < range.end; ++i)
		{
			/* divide 256 in 'diff' number
			of parts and process simultaneously */
			int chunk_size   = 256 / diff;
			double var_max   = -FLT_MAX;
			int    threshold = -1;
			if (i == range.end) {
				//remain chunk was dispatched by the last thread
				chunk_size += 256 % diff;
			}

			int start_index = i*chunk_size;
			int end_index   = start_index + chunk_size;

			for (int j = start_index; j < end_index; ++j) {
				int wB = 0;
				int wF = 0;
				double sumB = 0;
				double mB = 0;
				double mF = 0;
				double var_between = 0;

				for (int t = 0; t <= j; ++t) {
					wB += hist[t];                     // Weight Background
					sumB += t * hist[t];
				}
				if (wB == 0) continue;

				wF = 640 * 480 - wB;                    // Weight Foreground
				if (wF == 0) break;

				mB = sumB / (double)wB;                       // Mean Background
				mF = (total_hist_sum - sumB) / (double)wF;    // Mean Foregroun						
				var_between = (double)wB * (double)wF * (mB - mF) * (mB - mF);// Calculate Between Class Variance

				// Check if new maximum found
				if (var_between > var_max) {
					var_max = var_between;
					threshold = j;
				}
			}
			var_cand[i] = var_max;
			threshold_cand[i] = threshold;
		}
	}
};

class Parallel_process_apply_threshold : public cv::ParallelLoopBody
{

private:
	Mat img;
	Mat& retVal;
	int diff;
	int threshold_value;
	int type;
public:
	Parallel_process_apply_threshold(cv::Mat inputImgage, cv::Mat& outImage, int threshold_value, int type, int diffVal)
		: img(inputImgage), retVal(outImage), diff(diffVal), threshold_value(threshold_value), type(type) {}

	virtual void operator()(const cv::Range& range) const
	{

		//#pragma omp parallel for		
		for (int i = range.start; i < range.end; ++i)
		{
			/* divide image in 'diff' number
			of parts and process simultaneously */
			cv::Mat in(img, cv::Rect(0, (img.rows / diff)*i,
				img.cols, img.rows / diff));
			cv::Mat out(retVal, cv::Rect(0, (retVal.rows / diff)*i,
				retVal.cols, retVal.rows / diff));
			threshold(in, out, threshold_value, 255, type);
		}
	}
};

class Parallel_process_my_apply_threshold : public cv::ParallelLoopBody
{

private:
	Mat img;
	Mat& retVal;
	int diff;
	int threshold_value;
public:
	Parallel_process_my_apply_threshold(cv::Mat inputImgage, cv::Mat& outImage, int threshold_value, int diffVal)
		: img(inputImgage), retVal(outImage), diff(diffVal), threshold_value(threshold_value) {}

	virtual void operator()(const cv::Range& range) const
	{

		//#pragma omp parallel for		
		for (int i = range.start; i < range.end; ++i)
		{
			/* divide image in 'diff' number
			of parts and process simultaneously */
			cv::Mat in(img, cv::Rect(0, (img.rows / diff)*i,
				img.cols, img.rows / diff));
			cv::Mat out(retVal, cv::Rect(0, (retVal.rows / diff)*i,
				retVal.cols, retVal.rows / diff));
			
			for (int r = 0; r<in.rows; ++r) {
				for (int c = 0; c<in.cols; ++c) {
					if (in.at<uchar>(r, c) > threshold_value) {
						out.at<uchar>(r, c) = 255;
					}else {
						out.at<uchar>(r, c) = 0;
					}
				}
			}
		}
	}
};

inline void ParallelOtsu(cv::Mat inputImgage, cv::Mat& outImage, int type, int thread_num) {
	int *threshold_cand = new int[thread_num]();
	double *var_cand = new double[thread_num]();
	int *hist = new int[256]();
	int threshold_otsu = 0;
	double var_final   = -1;
	int total_hist_sum = 0;
	std::mutex mtx;

	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_hist(inputImgage, hist, mtx, total_hist_sum, thread_num));
	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_cal_thresh(hist, mtx, total_hist_sum, threshold_cand, var_cand, thread_num));

	for (int i = 0; i < thread_num; ++i) {
		if (var_final <= var_cand[i]) {
			var_final = var_cand[i];
			threshold_otsu = threshold_cand[i];
		}
	}

	cv::parallel_for_(cv::Range(0, thread_num), Parallel_process_apply_threshold(inputImgage, outImage, threshold_otsu, type, thread_num));

	delete [] var_cand;
	delete [] threshold_cand;
}

void SerialHist(cv::Mat inputImgage, cv::Mat& outImage, int *hist) {
	for (int r = 0; r<inputImgage.rows; ++r) {
		for (int c = 0; c<inputImgage.cols; ++c) {
			int temp = inputImgage.at<uchar>(r, c);
			hist[temp]++;
		}
	}
}

void SerialOtsu(cv::Mat inputImgage, int *hist, int total_hist_sum, int &threshold) {
	int wB = 0;
	int wF = 0;
	int total = 640 * 480;
	double sumB = 0;
	double varMax = -FLT_MAX;

	for (int t = 0; t<256; t++) {
		wB += hist[t];               // Weight Background
		if (wB == 0) continue;

		wF = total - wB;                 // Weight Foreground
		if (wF == 0) break;

		sumB += (float)(t * hist[t]);

		double mB = sumB / (double)wB;            // Mean Background
		double mF = (total_hist_sum - sumB) / (double)wF;    // Mean Foreground

										 // Calculate Between Class Variance
		double varBetween = (double)wB * (double)wF * (mB - mF) * (mB - mF);

		// Check if new maximum found
		if (varBetween > varMax) {
			varMax = varBetween;
			threshold = t;
		}
	}
}

#endif