/******************************************************************************
 *  Copyright (c) 2016, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *****************************************************************************/
/******************************************************************************
 *
 *
 * @file main_python.c
 *
 * Host code for BNN, overlay CNV-Pynq, to manage parameter loading, 
 * classification (inference) of single and multiple images
 * 
 *
 *****************************************************************************/
/*
	Generates data set at experiments/dataset and experiments/dataset-base

	Command avaliable:
	make clean-data
	./BNN (frame num) (expected class) (file numbering offset) (file numbering offset2 (for base files))
	./BNN 50 0 0 0  <- airplanes
	./BNN 50 1 50 10  <- automobile
	
*/

#include "../tiny_cnn/tiny_cnn.h"
#include "../tiny_cnn/util/util.h"
#include <iostream>
#include <fstream>
#include <string.h>
#include <chrono>
#include "foldedmv-offload.h"
#include <algorithm>
#include "opencv2/opencv.hpp"
#include <unistd.h>  		//for sleep
#include <omp.h>  		//for sleep
#include <sys/mman.h> //for clock
#include <sys/types.h> //for clock
#include <sys/stat.h>//for clock
#include <fcntl.h>//for clock
#include <stdio.h>//for clock
#include <stdlib.h>//for clock
//#include <opencv2/core/utility.hpp>

#include <main.hpp>
#include "load.hpp"
#include "roi_filter.hpp"
#include "win.hpp"
#include "uncertainty.hpp"


using namespace std;
using namespace tiny_cnn;
using namespace tiny_cnn::activation;
using namespace cv;
using namespace load;
using namespace basic;

#define frame_width 320		//176	//320	//640
#define frame_height 240		//144	//240	//480

#define HW_ADDR_GPIO 0xF8000170 //base: 0xF8000000 relative: 0x00000170 absolute: 0xF8000170 // ultrasclae+: 0xFF5E00C0
#define MAP_SIZE 4096UL
#define MAP_MASK (MAP_SIZE - 1)

float lambda;
unsigned int ok, failed; // used in FoldedMV.cpp

const std::string USER_DIR = "/home/xilinx/jose_bnn/bnn_lib_tests/";
const std::string BNN_PARAMS = USER_DIR + "params/cifar10/";

//main functions
int classify_frames(unsigned int no_of_frame, unsigned int expected_class, int offset, int base_offset);

template<typename T>
inline void print_vector(std::vector<T> &vec)
{
	std::cout << "{ ";
	for(auto const &elem : vec)
	{
		std::cout << elem << " ";
	}
	std::cout << "}" <<endl;
}

int main(int argc, char** argv)
{
	for(int i = 0; i < argc; i++)
		cout << "argv[" << i << "]" << " = " << argv[i] << endl;	
	
	unsigned int no_of_frame = atoi(argv[1]);
	unsigned int expected_class = (atoi(argv[2]));
	unsigned int offset = (atoi(argv[3]));
	unsigned int base_offset = (atoi(argv[4]));


	classify_frames(no_of_frame, expected_class, offset, base_offset);

	return 1;
}

int classify_frames(unsigned int no_of_frame, unsigned int expected_class, int offset, int base_offset){

    //Initialize variables
	cv::Mat cur_frame;
	unsigned int number_class = 10;
	vector<string> classes = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
	int frame_num = 0;
	int frames_dropped = 0;
	int processed_frames = base_offset;


	VideoCapture cap(0 + CV_CAP_V4L2);
	if(!cap.open(0))
	{
		cout << "cannot open camera" << endl;
	} 
	cap.set(CV_CAP_PROP_FRAME_WIDTH,frame_width);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT,frame_height);

	cout << "Get ready..." <<endl;

	while(frame_num < 100){
		cap >> cur_frame;
		imshow("Original", cur_frame);
		waitKey(25);
		frame_num++;
	}

	cout << "START NOW" <<endl;

	frame_num = 0;

    while(frame_num < no_of_frame){

		vector<double> u(5, 0.0);

		cap >> cur_frame;

		int num = offset + frame_num;

		imshow("Original", cur_frame);
		waitKey(25);

		// vector<int> compression_params;
		// compression_params.push_back( CV_IMWRITE_JPEG_QUALITY );
		// compression_params.push_back( 100 );
		string s = to_string(num);
		std::string img_path1 = "./experiments/dataset/" + s + "_" + classes[expected_class] + ".png";
		cv::imwrite(img_path1,cur_frame);

		if (frames_dropped == 4 || frame_num == 0){
			//store curframe to dataset_base folder as well
			string bs = to_string(processed_frames);
			std::string img_path2 = "./experiments/dataset-base/" + bs + "_" + classes[expected_class] + ".png";
			cv::imwrite(img_path2,cur_frame);

			frames_dropped = 0;
			processed_frames ++;

		} else {
			frames_dropped++;
		}

		frame_num++;

		char ESC = waitKey(1);	
		if (ESC == 27) 
        {
            cout << "ESC key is pressed by user" << endl;
            break;
        }	
    }
	cap.release();
    return 1;
}