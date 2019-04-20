#ifndef TBB_PARALLAL_SPEEDUP_H
#define TBB_PARALLAL_SPEEDUP_H
#pragma once

#include<opencv2\opencv.hpp>
using namespace cv;

class Parallel_process_gau : public cv::ParallelLoopBody
    {

    private:
        Mat img;
        Mat& retVal;
		Mat element;
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

#endif