#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

//non member function
const std::string TEST_DIR = "/home/xilinx/jose_bnn/bnn_lib_tests/experiments/";

namespace load
{
    vector<Mat> load_img(bool base){
        //load_png
        //Loading png images in test_images files
        vector<Mat> images;
        vector<cv::String> fn;

        if (base){
            glob(TEST_DIR + "dataset-base/*.png", fn, false);
        } else{
            glob(TEST_DIR + "dataset/*.png", fn, false);
        }
        size_t count = fn.size(); 
        for (size_t i=0; i<count; i++)
        {
            cout<<"loading images "<< i <<endl;
            images.push_back(imread(fn[i]));
        }
        return images;
    }
}
