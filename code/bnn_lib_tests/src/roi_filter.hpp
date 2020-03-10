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

    cv::Mat hsv_motion_mat;
    Rect small_bounding_r;


    Rect colour_seg(const Mat& cur, int low_thres, int up_thres);
    cv::Mat simple_optical_flow();
    void print_vector(std::vector<Point> &vec);
    int colour_similarity(Rect r, const Mat& a, int n);
    std::vector<float> normalise(std::vector<float> &cp);
    float entropy(std::vector<float> &arg_vec);
    

    public:

        int frame_width;
        int frame_height;
        cv::Mat cur_mat, cur_mat_grey, prev_mat, prev_mat_grey;

        Roi_filter(int w,int h){
            frame_width = w;
            frame_height = h;
            past_roi = Rect(Point(0,0), Point(w, h));
        }

        Rect naive_roi(const Mat& img, unsigned int roi_size);
        Rect basic_roi(const Mat& img);
        void init_enhanced_roi(const Mat& img);
        Rect enhanced_roi (const Mat& img);
        Rect get_past_roi();
        Rect get_full_roi();
        //void update_enhanced_roi_param(const Mat& img);
    
};

#endif
