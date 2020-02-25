#ifndef roi_filter
#define roi_filter
#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
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
    Rect past_roi;

    cv::Mat contour_map();
    cv::Mat dense_optical_flow(const Mat& contour_mat);
    cv::Mat weighted_map(const Mat& cur_mat);
    Rect colour_seg(const Mat& cur, int low_thres, int up_thres);
    void update_enhanced_roi_param (const Mat& motion_mat);
    cv::Mat simple_optical_flow();
    void print_vector(std::vector<Point> &vec);
    

    public:

        int frame_width;
        int frame_height;
        cv::Mat cur_mat, cur_mat_grey, prev_mat, prev_mat_grey, mask1, mask2, mask3;
        cv::Mat prev_img;
        //cv::Mat stored_mat;
        //cv::Mat stored_grey_mat;
        //vector<Point2f> stored_p0;

        Roi_filter(int w,int h){
            frame_width = w;
            frame_height = h;
            past_roi = Rect(Point(0,0), Point(w, h));
        }

        Rect naive_roi(const Mat& img, unsigned int roi_size);
        Rect basic_roi(const Mat& img, bool strict);
        void init_enhanced_roi(const Mat& img);
        Rect enhanced_roi (const Mat& img);
        Rect get_past_roi();
        void store_prev_img(const Mat& mat);
        void bitwise_and_roi(const Mat& mat);
    
};

#endif
