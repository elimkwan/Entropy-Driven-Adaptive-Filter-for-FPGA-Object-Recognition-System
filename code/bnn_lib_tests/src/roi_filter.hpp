#ifndef roi_filter
#define roi_filter
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

class Roi_filter{
    private:
    Rect expand_r(int x1, int y1, int x2, int y2, float p);
    //bool compareContourAreas ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 );

    public:
        // Mat* src;
        // Mat* dest;
        // int frame_size;

        int frame_width;
        int frame_height;

        Roi_filter(int w,int h){
            frame_width = w;
            frame_height = h;
        }

        Rect naive_roi(const Mat& img, unsigned int roi_size);
        Rect real_roi(const Mat& img, unsigned int roi_size);
    
};

#endif
