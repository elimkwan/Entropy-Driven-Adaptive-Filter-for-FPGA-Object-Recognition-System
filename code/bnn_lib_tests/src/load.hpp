#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

//non member function
const std::string TEST_DIR = "/home/xilinx/jose_bnn/bnn_lib_tests/experiments/";

namespace load
{
    vector<Mat> load_img(){
        //load_png
        //Loading png images in test_images files
        vector<Mat> images;
        vector<cv::String> fn;
        glob(TEST_DIR + "images/*.jpg", fn, false);
        size_t count = fn.size(); 
        for (size_t i=0; i<count; i++)
        {
            cout<<"loading images "<< i <<endl;
            images.push_back(imread(fn[i]));
        }
        return images;
    }

    // void load_video(){

    // }

    // VideoCapture cam_init(){
    //     VideoCapture cap(0 + CV_CAP_V4L2);
	//     if(!cap.open(0))
	//     {
	//         cout << "cannot open camera" << endl;
	//         return 0;
	//     }

	//     cap.set(CV_CAP_PROP_FRAME_WIDTH,frame_width);
	//     cap.set(CV_CAP_PROP_FRAME_HEIGHT,frame_height);
	//     std::cout << "\nCamera resolution = " << cap.get(CV_CAP_PROP_FRAME_WIDTH) << "x" << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << std::endl;
    //     return cap;
    // }

    // VideoCapture use_cam(VideoCapture cap){
    //     Mat cap_frame;
    //     cap >> cap_frame;
	// 	return cap_frame;
    // }
}
