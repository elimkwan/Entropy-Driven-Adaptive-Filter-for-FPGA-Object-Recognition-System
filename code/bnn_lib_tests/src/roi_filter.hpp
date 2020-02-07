#ifndef roi_filter
#define roi_filter
#include "opencv2/opencv.hpp"
#include <iostream>
// #include "opencv2/core.hpp"
// #include <opencv2/highgui.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/videoio.hpp>
// #include <opencv2/video.hpp>

using namespace cv;
using namespace std;

class Roi_filter{
    private:
    Rect expand_r(int x1, int y1, int x2, int y2, float p);

    cv::Mat contour_map();
    cv::Mat dense_optical_flow(const Mat& contour_mat);
    cv::Mat weighted_map(const Mat& cur_mat);
    Rect bounding_rect(const Mat& cur_mat);
    void update_enhanced_roi_param (const Mat& motion_mat);
    

    public:

        int frame_width;
        int frame_height;
        cv::Mat cur_mat, cur_mat_grey, prev_mat, prev_mat_grey, mask1, mask2, mask3;

        //cv::Mat stored_mat;
        //cv::Mat stored_grey_mat;
        //vector<Point2f> stored_p0;

        Roi_filter(int w,int h){
            frame_width = w;
            frame_height = h;
        }

        Rect naive_roi(const Mat& img, unsigned int roi_size);
        Rect basic_roi(const Mat& img);
        void init_enhanced_roi(const Mat& img);
        Rect enhanced_roi (const Mat& img);
    
};

#endif
