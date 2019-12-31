#ifndef roi_filter
#define roi_filter
#include "opencv2/opencv.hpp"

using namespace cv;

class Roi_filter{
    public:
        // Mat* src;
        // Mat* dest;
        // int frame_size;

        // Roi_filter(){
        //     src = 0x0;
        //     dest = 0x0;
        // }

        cv::Mat cur_to_src(Mat img, unsigned int frame_size);
        void detection();
    
};

#endif
