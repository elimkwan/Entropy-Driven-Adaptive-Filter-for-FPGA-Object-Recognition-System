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
//#include <opencv2/core/utility.hpp>

#include "load.hpp"
#include "roi_filter.hpp"
#include "win.hpp"


using namespace std;
using namespace tiny_cnn;
using namespace tiny_cnn::activation;
using namespace cv;
using namespace load;

#define frame_width 320		//176	//320	//640
#define frame_height 240		//144	//240	//480

float lambda;
unsigned int ok, failed; // used in FoldedMV.cpp

const std::string USER_DIR = "/home/xilinx/jose_bnn/bnn_lib_tests/";
const std::string BNN_PARAMS = USER_DIR + "params/cifar10/";
//const std::string TEST_DIR = "/home/xilinx/jose_bnn/bnn_lib_tests/experiments/";

ofstream myfile;

//main functions
int classify_frames(std::string in_type, unsigned int frame_num, unsigned int frame_size, unsigned int win_step, unsigned int win_length, bool roi, unsigned int expected_class);

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

double clockToMilliseconds(clock_t ticks){
    // units/(units/time) => time (seconds) * 1000 = milliseconds
    return (ticks/(double)CLOCKS_PER_SEC)*1000.0;
}


template<typename T>
float expDecay(T lambda, int t, int N = 1)
{
	// Remove N if it is not needed
	return N * std::exp(-(lambda * (T)t));
}


// Convert matrix into a vector 
// |1 0 0|
// |0 1 0| -> [1 0 0 0 1 0 0 0 1]
// |0 0 1|
template<typename T>
void flatten_mat(cv::Mat &m, std::vector<T> &v)
{
	if(m.isContinuous()) 
	{
		//cout<< "data is continuous"<< endl;
		v.assign(m.datastart, m.dataend);
	} 
	else 
	{
		cout<< "data is not continuous"<< endl;
		for (int i = 0; i < m.rows; ++i) 
		{
			v.insert(v.end(), m.ptr<T>(i), m.ptr<T>(i)+m.cols);
		}
	}
}

template<typename T>
inline std::vector<float> normalise(std::vector<T> &vec)
{	
	std::vector<float> cp(vec.begin(), vec.end());
	T mx = *max_element(std::begin(cp), std::end(cp));
	
	for(auto &elem : cp)
		elem = (float)elem / mx;
	
	return cp;
}

/*
	Calculate certainty

	@para arg_vec: input vector with floating points
	@return vector [e^(class1 probability)/sum, e^(class2 probability)/sum... e^(class10 probability)/sum], where sum = summation of e^(class probability) of all the classes
*/
template<typename T>
vector<float> calculate_certainty(std::vector<T> &arg_vec)
{
	// Normalise the vector
	std::vector<float> norm_vec = normalise(arg_vec);
	float mx = *max_element(std::begin(norm_vec), std::end(norm_vec));
	float sum = 0;
	for(auto const &elem : norm_vec)
		sum += exp(elem);
	
	if(sum == 0){
		std::cout << "Division by zero, sum = 0" << std::endl;
	}
	// Try to use OpenMP
	for(int i=0; i<10;i++)
	{
		norm_vec[i] = exp(norm_vec[i]) / sum;
	}

	return norm_vec;
}



void makeNetwork(network<mse, adagrad> & nn) {
  nn
#ifdef OFFLOAD
      << chaninterleave_layer<identity>(3, 32*32, false)
      << offloaded_layer(3*32*32, 10, &FixedFoldedMVOffload<8, 1>, 0xdeadbeef, 0)
#endif
      ;
}


extern "C" void load_parameters(const char* path)
{
#include "config.h"
FoldedMVInit("cnv-pynq");
network<mse, adagrad> nn;
makeNetwork(nn);
        cout << "Setting network weights and thresholds in accelerator..." << endl;
        FoldedMVLoadLayerMem(path , 0, L0_PE, L0_WMEM, L0_TMEM);
        FoldedMVLoadLayerMem(path , 1, L1_PE, L1_WMEM, L1_TMEM);
        FoldedMVLoadLayerMem(path , 2, L2_PE, L2_WMEM, L2_TMEM);
        FoldedMVLoadLayerMem(path , 3, L3_PE, L3_WMEM, L3_TMEM);
        FoldedMVLoadLayerMem(path , 4, L4_PE, L4_WMEM, L4_TMEM);
        FoldedMVLoadLayerMem(path , 5, L5_PE, L5_WMEM, L5_TMEM);
        FoldedMVLoadLayerMem(path , 6, L6_PE, L6_WMEM, L6_TMEM);
        FoldedMVLoadLayerMem(path , 7, L7_PE, L7_WMEM, L7_TMEM);
        FoldedMVLoadLayerMem(path , 8, L8_PE, L8_WMEM, L8_TMEM);
}

extern "C" unsigned int inference(const char* path, unsigned int results[64], int number_class, float *usecPerImage)
{

FoldedMVInit("cnv-pynq");

network<mse, adagrad> nn;

makeNetwork(nn);
std::vector<label_t> test_labels;
std::vector<vec_t> test_images;

parse_cifar10(path, &test_images, &test_labels, -1.0, 1.0, 0, 0);
std::vector<unsigned int> class_result;
float usecPerImage_int;
class_result=testPrebuiltCIFAR10_from_image<8, 16>(test_images, number_class, usecPerImage_int);
if(results)
	std::copy(class_result.begin(),class_result.end(), results);
if (usecPerImage)
    *usecPerImage = usecPerImage_int;
return (std::distance(class_result.begin(),std::max_element(class_result.begin(), class_result.end())));
}

extern "C" unsigned int inference_test(const char* path, unsigned int results[64], int number_class, float *usecPerImage,unsigned int img_num)
{

FoldedMVInit("cnv-pynq");

network<mse, adagrad> nn;

makeNetwork(nn);
std::vector<label_t> test_labels;
std::vector<vec_t> test_images;

parse_cifar10(path, &test_images, &test_labels, -1.0, 1.0, 0, 0);
float usecPerImage_int;

testPrebuiltCIFAR10<8, 16>(test_images, test_labels, number_class,img_num);


}

extern "C" unsigned int* inference_multiple(const char* path, int number_class, int *image_number, float *usecPerImage, unsigned int enable_detail = 0)
{

FoldedMVInit("cnv-pynq");

network<mse, adagrad> nn;

makeNetwork(nn);

std::vector<label_t> test_labels;
std::vector<vec_t> test_images;

parse_cifar10(path,&test_images, &test_labels, -1.0, 1.0, 0, 0);
std::vector<unsigned int> all_result;
std::vector<unsigned int> detailed_results;
float usecPerImage_int;
all_result=testPrebuiltCIFAR10_multiple_images<8, 16>(test_images, number_class, detailed_results, usecPerImage_int);
unsigned int * result;
if (image_number)
   *image_number = all_result.size();
if (usecPerImage)
    *usecPerImage = usecPerImage_int;
if (enable_detail)
{
	result = new unsigned int [detailed_results.size()];
	std::copy(detailed_results.begin(),detailed_results.end(), result);
}
else
{
	result = new unsigned int [all_result.size()];
	std::copy(all_result.begin(),all_result.end(), result);
}
   
return result;
}

extern "C" void free_results(unsigned int * result)
{
delete[] result;
}

extern "C" void deinit() {
FoldedMVDeinit();
}


/*
	Command avaliable:
	//input src, number of frames to be run, display window size, window step, window length, with roi filter, expected class
	./BNN cam 1000 128 5 12 1 1
	./BNN pics 1000 128 5 12 1 1
	./BNN video 1000 128 5 12 1 1 (not yet)
*/
int main(int argc, char** argv)
{
	for(int i = 0; i < argc; i++)
		cout << "argv[" << i << "]" << " = " << argv[i] << endl;	
	
	std::string in_type = argv[1];
	unsigned int no_of_frame = atoi(argv[2]);
	unsigned int frame_size = atoi(argv[3]);
	unsigned int win_step = atoi(argv[4]);
	unsigned int win_length = atoi(argv[5]);
	bool with_roi = (atoi(argv[6]) > 0);
	unsigned int expected_class = (atoi(argv[7]));

	classify_frames(in_type, no_of_frame, frame_size, win_step, win_length, with_roi, expected_class);

	return 1;
}

int classify_frames(std::string in_type, unsigned int no_of_frame, unsigned int frame_size, unsigned int win_step, unsigned int win_length, bool with_roi, unsigned int expected_class){

    //Initialize variables
	cv::Mat reduced_sized_frame(32, 32, CV_8UC3);
	cv::Mat cur_frame, src;
	Mat bnn_input = Mat(frame_size, frame_size, CV_8UC3);
	float_t scale_min = -1.0;
    float_t scale_max = 1.0;
	unsigned int number_class = 10;
	unsigned int output = 0;
	vector<string> classes = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
    unsigned int size, frame_num = 0;
	tiny_cnn::vec_t outTest(number_class, 0);
	const unsigned int count = 1;
	std::vector<uint8_t> bgr;

    // Initialize the network 
    deinit();
	load_parameters(BNN_PARAMS.c_str()); 
	printf("Done loading BNN\n");
	FoldedMVInit("cnv-pynq");
	network<mse, adagrad> nn;
	makeNetwork(nn);

    //Allocate memories
    // # of ExtMemWords per input
	const unsigned int psi = 384; //paddedSize(imgs.size()*inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
	// # of ExtMemWords per output
	const unsigned int pso = 16; //paddedSize(64*outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
	if(INPUT_BUF_ENTRIES < psi)
	throw "Not enough space in accelBufIn";
	if(OUTPUT_BUF_ENTRIES < pso)
	throw "Not enough space in accelBufOut";
	// allocate host-side buffers for packed input and outputs
	ExtMemWord * packedImages = (ExtMemWord *)sds_alloc((count * psi)*sizeof(ExtMemWord));
	ExtMemWord * packedOut = (ExtMemWord *)sds_alloc((count * pso)*sizeof(ExtMemWord));

    vector<Mat> frames;

	VideoCapture cap(0 + CV_CAP_V4L2);
	if(!cap.open(0))
	{
		cout << "cannot open camera" << endl;
	} 
	cap.set(CV_CAP_PROP_FRAME_WIDTH,frame_size);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT,frame_size);
	//std::cout << "\nCamera resolution = " << cap.get(CV_CAP_PROP_FRAME_WIDTH) << "x" << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << std::endl;
	size = no_of_frame;

    //Load input: from pictures, video or webcam to set curframe
    if (in_type == "pics"){
        //load png
		cap.release();
        frames = load::load_img();
        size = frames.size();
    } else if (in_type == "video"){
		cap.release();
        //load video
        //size = no. of frame in the video
    }

	Roi_filter r_filter;

    while(frame_num < size){
        // Data preprocessing: transform cur_frame to src then to bnn_input
        if (in_type == "pics"){
            cur_frame = frames[frame_num];
        } 
		else {
			cap >> cur_frame;
        }
        src = r_filter.cur_to_src(cur_frame, frame_size);

        //Resizing frame for bnn
        cv::resize(src, bnn_input, cv::Size(frame_size, frame_size), 0, 0, cv::INTER_CUBIC );
        cv::resize(bnn_input, reduced_sized_frame, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC );			
        flatten_mat(reduced_sized_frame, bgr);
        vec_t img;
        std::transform(bgr.begin(), bgr.end(), std::back_inserter(img),[=](unsigned char c) { return scale_min + (scale_max - scale_min) * c / 255; });
        quantiseAndPack<8, 1>(img, &packedImages[0], psi);
    
        // Call the hardware function
		kernelbnn((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, false, 0, 0, 0, 0, count,psi,pso,1,0);
		if (frame_num != 1)
		{
			kernelbnn((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, false, 0, 0, 0, 0, count,psi,pso,0,1);
		}
		// Extract the output of BNN and classify result
		std::vector<unsigned int> class_result;
		copyFromLowPrecBuffer<unsigned short>(&packedOut[0], outTest);
		for(unsigned int j = 0; j < number_class; j++) {			
			class_result.push_back(outTest[j]);
		}		
		output = distance(class_result.begin(),max_element(class_result.begin(), class_result.end()));

		cout << "output: " << output << endl;
        //Data post-processing:
        //output_filter(output);
		results_history.insert(results_history.begin(), calculate_certainty(class_result));
		if (results_history.size() > win_length)
		{
			results_history.pop_back();
		}
		int adjusted_output = 0;
		adjusted_output = output_filter(win_step, step_counts, past_output, results_history, weights);
		if (step_counts < win_step) {
			step_counts++;
		} else {
			step_counts = 0;
			past_output = adjusted_output;
		}

		//Display output
		if (in_type == "pics"){
			//imshow("Original", cur_frame);
			vector<int> compression_params;
			compression_params.push_back( CV_IMWRITE_JPEG_QUALITY );
			compression_params.push_back( 100 );
			std::string img_path = "../experiments/results/" + classes[output] + ".jpg";
			imwrite(img_path,cur_frame, compression_params);
			//waitKey(0);
		} else {
			putText(cur_frame, classes[output], Point(15, 55), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));	
			imshow("Original", cur_frame);
			waitKey(25);	
		}

        frame_num++;
    }

	if (cap.open(0)){
		cap.release();
	}

    //Release memory
    sds_free(packedImages);
	sds_free(packedOut);
    return 1;
}
