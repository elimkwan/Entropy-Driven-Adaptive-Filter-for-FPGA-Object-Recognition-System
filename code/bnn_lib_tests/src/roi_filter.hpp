#ifndef roi_filter
#define roi_filter
#include "opencv2/opencv.hpp"

using namespace cv;

class Roi_filter{
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
        Rect real_roi(const Mat& img, unsigned int roi_size)
    
};

#endif
