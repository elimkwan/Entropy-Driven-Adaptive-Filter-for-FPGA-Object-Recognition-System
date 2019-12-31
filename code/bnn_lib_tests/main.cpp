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
#include "tiny_cnn/tiny_cnn.h"
#include "tiny_cnn/util/util.h"
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


using namespace std;
using namespace tiny_cnn;
using namespace tiny_cnn::activation;
using namespace cv;

#define frame_width 320		//176	//320	//640
#define frame_height 240		//144	//240	//480

float lambda;
unsigned int ok, failed; // used in FoldedMV.cpp

const std::string USER_DIR = "/home/xilinx/jose_bnn/bnn_lib_tests/";
const std::string BNN_PARAMS = USER_DIR + "params/cifar10/";

ofstream myfile;

//main functions
int classify_frames_cam_sw(int frames, int win_height, int win_width, int win_step, int win_length, int argv7);
int classify_frames_load(int win_height, int win_width);
int classify_frames_load_roi_test(int win_height, int win_width, int roi_count);
int classify_frames_load_roi(int win_height, int win_width);

//helper and lambda function
void helpMessage(int argc, char** argv);
int output_filter(int arg_win_step, int arg_count, int arg_past_output, std::vector<std::vector<float>> arg_results_history, std::vector<float> arg_weights);


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


void helpMessage(int argc, char** argv)
{
	//TODO: to be updated
	cout << argv[1] << " <mode> " << endl;
	cout << "mode = hw, sw" << endl;
	cout << "hw: Sowftware parts run in parallel with hw part (async/wait)" << endl;
	cout << "sw: Multithreading software functions and run in parallel with hw part " << endl;
	cout << argv[2] << " The expected output class number based on CIFAR-10 [0 to 9] " << endl;
	cout << argv[3] << " Number of frames to automatically stop the program" << endl;
	cout << argv[4] << " BNN input frame size -- only for sw mode, 0 for full window" << endl;
}

/*
	Smoothening out output

	@para arg_win_step:
	@para arg_count;
	@para arg_results_history:
	@para arg_weights
	@return output
*/
int output_filter(int arg_win_step, int arg_count, int arg_past_output, std::vector<std::vector<float>> arg_results_history, std::vector<float> arg_weights){  	

	cout << "output filer starts here ..." << endl;

	if (arg_results_history.size() <= arg_win_step){
		//not enough data for previous analysis, return real time data
		cout << "outputing real time data" << endl;
		std::cout << "arg_count: " << arg_count << std::endl;
		std::cout << "arg_results_history" << std::endl;
		for (int i = 0; i < arg_results_history.size(); i++)
		{
			print_vector(arg_results_history[i]);
		}

		//std::vector<float> current_result = arg_results_history[arg_count];
		int result_index = distance(arg_results_history[0].begin(), max_element(arg_results_history[0].begin(), arg_results_history[0].end()));
		//float result_value = 0.00f;
		//result_value = std::max_element(current_result.begin(), current_result.end());
		std::cout << "real time data results index: " << result_index << std::endl;
		print_vector(arg_results_history[0]);
		return result_index;
	}

	if (arg_count < arg_win_step){
		return arg_past_output;

	}else if ( arg_count == arg_win_step){

		cout << "outputing current analysised result" << endl;
		std::vector<float> adjusted_results(10, 0);
		for(int i = 0; i < arg_results_history.size(); i++)
		{ 
			for(int j = 0; j < 10; j++)
			{
				adjusted_results[j] += (arg_weights[i] * arg_results_history[i][j]);
			}
		}
		return distance(adjusted_results.begin(), max_element(adjusted_results.begin(), adjusted_results.end()));
	}

}

int roi_detection (cv::Mat cur_mat)
{
	cv::Mat blur_mat, grey_mat, sobel_mat, thres_mat, canny_mat, contour_mat;
	contour_mat = cur_mat.clone();
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	
	GaussianBlur(cur_mat, cur_mat, Size(3,3), 0, 0, BORDER_DEFAULT );
	cv::cvtColor(cur_mat, grey_mat, CV_BGR2GRAY);

	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	/// Gradient X
	Sobel( grey_mat, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	/// Gradient Y
	Sobel( grey_mat, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );
	convertScaleAbs( grad_y, abs_grad_y );
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobel_mat);

	int thresh = 100;
	Canny( sobel_mat, canny_mat, thresh, thresh*2 );

	opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel);

	Rect bounding_rect;
	vector< cv::Point> cnts;
	vector< vector< cv::Point> > contours;
	vector< vector< cv::Point> > contours_filtered;
	findContours(canny_mat, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	for (size_t i = 0; i< contours.size(); i++) // iterate through each contour.
    {
        double area = contourArea(contours[i]);  //  Find the area of contour

        if (area > 100)
        {
            contours_filtered.push_back (contours[i]);
        }
    }
	//cnts = np.concatenate(contours_filtered);
	//bounding_rect = boundingRect(cnts);

	imshow( "sobel", sobel_mat);
	waitKey(0);
	imshow( "canny", canny_mat);
	waitKey(0);


	// double thresh = 127;
	// double maxValue = 255; 
	// threshold(grey_mat,thres_mat, thresh, maxValue, THRESH_BINARY_INV);

	// Canny(thres_mat, canny_mat, 30, 128, 3, false);
	// vector< vector< cv::Point> > contours;
	// findContours(canny_mat, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// int largest_area = 0;
    // int largest_contour_index = 0;
    // Rect bounding_rect;
	// for (size_t i = 0; i< contours.size(); i++) // iterate through each contour.
    // {
    //     double area = contourArea(contours[i]);  //  Find the area of contour

    //     if (area > largest_area)
    //     {
    //         largest_area = area;
    //         largest_contour_index = i;               //Store the index of largest contour
    //         bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
    //     }
    // }
	// drawContours(contour_mat, contours, largest_contour_index, Scalar(0, 255, 0), 2);

	// imshow("greyed image", grey_mat);
	// waitKey(0);
	// imshow("thres image", thres_mat);
	// waitKey(0);
	// imshow("canny image", canny_mat);
	// waitKey(0);
	// imshow("roi image", contour_mat);
	// waitKey(0);

	//rectangle(cur_frame, Point(x1, y1), Point(x2, y2), colour, 2);
	return 1;
}


/*
	Command avaliable:
	./BNN win_exp 5 12 1
	./BNN cam_sw 128 1000 5 12 1 //operation, display window size, number of frames to be run, window step, window length, expected class
	./BNN load 128 //operation, display window size
	./BNN load_roi_test 300 150 //operations, window size, ROI step size in both x,y directions
	./BNN load_roi 128 //operation canny, display window size
*/
int main(int argc, char** argv)
{
	for(int i = 0; i < argc; i++)
		cout << "argv[" << i << "]" << " = " << argv[i] << endl;	
	
	if (argc >= 3){
		std::string operation = argv[1];
		int win_height = atoi(argv[2]);
		int win_width = atoi(argv[2]);
		
		if (argc >= 6)
		{
			//cam_sw, cam_hw
			int frames = atoi(argv[3]);
			int win_step =  atoi(argv[4]);
			int win_length = atoi(argv[5]);

			if (operation == "cam_sw")
			{
				int expected_class_num = 1;
				expected_class_num = atoi(argv[6]);
				classify_frames_cam_sw(frames, win_height, win_width, win_step, win_length, expected_class_num);
				return 1;
			}
			
			cout << "argc = " << argc << endl;
			helpMessage(argc, argv);
			return -1;
			
		}
		else if (argc >= 3)
		{
			//load, load-roi,test
			if (operation == "load")
			{
				classify_frames_load(win_height, win_width);
				return 1;
			}
			else if (operation == "load_roi")
			{
				classify_frames_load_roi(win_height, win_width);
				return 1;
			}
			else if (operation == "load_roi_test" && argc >= 4)
			{
				int roi_steps = atoi(argv[3]);
				classify_frames_load_roi_test(win_height, win_width, roi_steps);
				return 1;
			}
			
			cout << "argc = " << argc << endl;
			helpMessage(argc, argv);
			return -1;
		}
	}

	cout << "argc = " << argc << endl;
	helpMessage(argc, argv);
	return -1;	
}

/*
	load all the jpg images in ./test_images for testing, implementing ROI detection
	output result to test_results

*/
int classify_frames_load_roi(int win_height, int win_width)
{
	//initialize variables
	cv::Mat reduced_sized_frame(32, 32, CV_8UC3);
	cv::Mat cur_frame;
	Mat bnn_input = Mat(win_width, win_height, CV_8UC3);
	float_t scale_min = -1.0;
    float_t scale_max = 1.0;
	unsigned int frame_num = 0;	
	int number_class = 10;
	int output = 0;
	


	myfile.open ("result_load.csv",std::ios_base::app);
	printf("Hello BNN\n");

	deinit();
	load_parameters(BNN_PARAMS.c_str()); 
	printf("Done loading BNN\n");
	
	// Initialize the network 
	FoldedMVInit("cnv-pynq");
	network<mse, adagrad> nn;
	makeNetwork(nn);

	// Get a list of all the output classes
	vector<string> classes;
	ifstream file((USER_DIR + "params/cifar10/classes.txt").c_str());
	cout << "Opening parameters at: " << (USER_DIR + "params/cifar10/classes.txt") << endl;
	string str;
	if (file.is_open())
	{
		cout << "Classes: [";
		while (getline(file, str))
		{
			cout << str << ", "; 
			classes.push_back(str);
		}
		cout << "]" << endl;
		
		file.close();
	}
	else
	{
		cout << "Failed to open classes.txt" << endl;
	}

	//Loading png images in test_images files
	vector<cv::String> fn;
	glob("test_images/*.jpg", fn, false);

	vector<Mat> images;
	size_t count = fn.size(); 
	for (size_t i=0; i<count; i++)
	{
		cout<<"loading images "<< i <<endl;
		images.push_back(imread(fn[i]));
		//std::string window_name = "Display" + std::to_string(i); //to display the image
		//imshow(window_name, images[i]);
		//waitKey(0);
	}

	// # of ExtMemWords per input
	const unsigned int psi = 384; //paddedSize(imgs.size()*inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
	// # of ExtMemWords per output
	const unsigned int pso = 16; //paddedSize(64*outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
	if(INPUT_BUF_ENTRIES < psi)
	throw "Not enough space in accelBufIn";
	if(OUTPUT_BUF_ENTRIES < pso)
	throw "Not enough space in accelBufOut";
	// allocate host-side buffers for packed input and outputs
	//ExtMemWord * packedImages = new ExtMemWord[(count * psi)];
	ExtMemWord * packedImages = (ExtMemWord *)sds_alloc((count * psi)*sizeof(ExtMemWord));
	//ExtMemWord * packedOut = new ExtMemWord[(count * pso)];
	ExtMemWord * packedOut = (ExtMemWord *)sds_alloc((count * pso)*sizeof(ExtMemWord));


	for (size_t i=0; i<count; i++){
		cur_frame = images[i];
		int img_height = cur_frame.size().height;
		int img_width = cur_frame.size().width;
		//Mat bnn_input = Mat(cv::Size(win_width, win_height));
		std::vector<uint8_t> bgr;

		//detecting ROI
		cout << "trying to detect ROI" << endl;
		roi_detection(cur_frame);

		if(win_height == 0)
		{
			cv::resize(cur_frame, reduced_sized_frame, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC );	
			flatten_mat(reduced_sized_frame, bgr);			
			vec_t img;
			std::transform(bgr.begin(), bgr.end(), std::back_inserter(img),[=](unsigned char c) { return scale_min + (scale_max - scale_min) * c / 255; });
			quantiseAndPack<8, 1>(img, &packedImages[0], psi);										
		} else {	
			// Take only part of the frame from the original frame (center)
			Rect R(Point((img_width/2)-(win_width/2), (img_height/2)-(win_height/2)), Point((img_width/2)+(win_width/2), (img_height/2)+(win_height/2)));

			cv::resize(cur_frame(R), bnn_input, cv::Size(win_width, win_height), 0, 0, cv::INTER_CUBIC );
			cv::resize(bnn_input, reduced_sized_frame, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC );			
			flatten_mat(reduced_sized_frame, bgr);
			vec_t img;
			std::transform(bgr.begin(), bgr.end(), std::back_inserter(img),[=](unsigned char c) { return scale_min + (scale_max - scale_min) * c / 255; });
			quantiseAndPack<8, 1>(img, &packedImages[0], psi);															
		}

		// Call the hardware function
		kernelbnn((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, false, 0, 0, 0, 0, count,psi,pso,1,0);
		if (frame_num != 1)
		{
			kernelbnn((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, false, 0, 0, 0, 0, count,psi,pso,0,1);
		}
		// Extract the output of BNN and classify result
		std::vector<unsigned int> class_result;
		tiny_cnn::vec_t outTest(number_class, 0);
		copyFromLowPrecBuffer<unsigned short>(&packedOut[0], outTest);
		for(unsigned int j = 0; j < number_class; j++) {			
			class_result.push_back(outTest[j]);
		}		
		output = distance(class_result.begin(),max_element(class_result.begin(), class_result.end()));

		cout<< "classification result: " << classes[output] << endl;


		std::string expected_class = fn[i];
		expected_class.erase(0,12);
		std:string img_name = expected_class;
		expected_class.erase(expected_class.find_last_of('.'));
		expected_class.erase(std::remove_if(std::begin(expected_class), std::end(expected_class),[](char ch) { return std::isdigit(ch); }), expected_class.end());

		if (expected_class == classes[output])
		{
			cout<<"correctly identified "<< expected_class <<endl;
		} else{
			cout<<"cannot identify it is a " << expected_class <<endl;
		}

		putText(cur_frame, classes[output], Point((img_width/2)-(win_width/2) + 10, (img_height/2)-(win_height/2)+10), FONT_HERSHEY_PLAIN, 1 , Scalar(0, 255, 0));
		rectangle(cur_frame, Point((img_width/2)-(win_width/2), (img_height/2)-(win_height/2)), Point((img_width/2)+(win_width/2), (img_height/2)+(win_height/2)), Scalar(0, 0, 255)); // draw a 32x32 box at the centre
		imshow(img_name, cur_frame);

		vector<int> compression_params;
		compression_params.push_back( CV_IMWRITE_JPEG_QUALITY );
		compression_params.push_back( 100 );
		std::string img_path = "./test_results/" + img_name;
		imwrite(img_path,cur_frame, compression_params);

		waitKey(0);
	}

	sds_free(packedImages);
	sds_free(packedOut);
	return 0;
}

int classify_frames_load_roi_test(int win_height, int win_width, int roi_steps)
{
	//initialize variables
	cv::Mat reduced_sized_frame(32, 32, CV_8UC3);
	cv::Mat cur_frame;
	Mat bnn_input = Mat(win_width, win_height, CV_8UC3);
	float_t scale_min = -1.0;
    float_t scale_max = 1.0;
	unsigned int frame_num = 0;	
	int number_class = 10;
	int output = 0;
	
	myfile.open ("result_load.csv",std::ios_base::app);
	printf("Hello BNN\n");

	deinit();
	load_parameters(BNN_PARAMS.c_str()); 
	printf("Done loading BNN\n");
	
	// Initialize the network 
	FoldedMVInit("cnv-pynq");
	network<mse, adagrad> nn;
	makeNetwork(nn);

	// Get a list of all the output classes
	vector<string> classes;
	ifstream file((USER_DIR + "params/cifar10/classes.txt").c_str());
	cout << "Opening parameters at: " << (USER_DIR + "params/cifar10/classes.txt") << endl;
	string str;
	if (file.is_open())
	{
		cout << "Classes: [";
		while (getline(file, str))
		{
			cout << str << ", "; 
			classes.push_back(str);
		}
		cout << "]" << endl;
		
		file.close();
	}
	else
	{
		cout << "Failed to open classes.txt" << endl;
	}

	//Loading png images in test_images files
	vector<cv::String> fn;
	glob("test_images/*.jpg", fn, false);

	vector<Mat> images;
	size_t count = fn.size(); 
	for (size_t i=0; i<count; i++)
	{
		cout<<"loading images "<< i <<endl;
		images.push_back(imread(fn[i]));
	}

	// # of ExtMemWords per input
	const unsigned int psi = 384; //paddedSize(imgs.size()*inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
	// # of ExtMemWords per output
	const unsigned int pso = 16; //paddedSize(64*outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
	if(INPUT_BUF_ENTRIES < psi)
	throw "Not enough space in accelBufIn";
	if(OUTPUT_BUF_ENTRIES < pso)
	throw "Not enough space in accelBufOut";
	// allocate host-side buffers for packed input and outputs
	//ExtMemWord * packedImages = new ExtMemWord[(count * psi)];
	ExtMemWord * packedImages = (ExtMemWord *)sds_alloc((count * psi)*sizeof(ExtMemWord));
	//ExtMemWord * packedOut = new ExtMemWord[(count * pso)];
	ExtMemWord * packedOut = (ExtMemWord *)sds_alloc((count * pso)*sizeof(ExtMemWord));


	for (size_t i=0; i<count; i++){
		cur_frame = images[i];
		int img_height = cur_frame.size().height;
		int img_width = cur_frame.size().width;
		std::vector<uint8_t> bgr;
		int dist_x = 0, dist_y = 0; 
		int x1=0, y1=0, x2=0, y2=0;
		cv::Scalar colour = cv::Scalar(0, 255, 0);

		y1 = dist_y;
		y2 = dist_y + win_height;

		while(true)
		{
			while(true)
			{
				x1 = dist_x;
				x2 = dist_x + win_width;

				if (x2>img_width){
					break;
				}

				//Take only part of the frame from the original frame (center)
				Rect R(Point(x1, y1), Point(x2, y2));
				cv::resize(cur_frame(R), bnn_input, cv::Size(win_width, win_height), 0, 0, cv::INTER_CUBIC );
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
				tiny_cnn::vec_t outTest(number_class, 0);
				copyFromLowPrecBuffer<unsigned short>(&packedOut[0], outTest);
				for(unsigned int j = 0; j < number_class; j++) {			
					class_result.push_back(outTest[j]);
				}		
				output = distance(class_result.begin(),max_element(class_result.begin(), class_result.end()));

				putText(cur_frame, classes[output], Point(x1+5, y1+15), FONT_HERSHEY_PLAIN, 1 , colour);
				rectangle(cur_frame, Point(x1, y1), Point(x2, y2), colour, 2); // draw a 32x32 box at the centre

				//alternate colours to produce clearer results
				if (colour == cv::Scalar(0, 255, 0)){
					colour = cv::Scalar(0, 0, 255);
				} else {
					colour = cv::Scalar(0, 255, 0);
				}

				dist_x += roi_steps;
			}

			dist_x = 0;
			dist_y += roi_steps;

			x1 = dist_x;
			y1 = dist_y;
			x2 = dist_x + win_width;
			y2 = dist_y + win_height;

			if (y2>img_height){
				break;
			}
		}

		std::string expected_class = fn[i];
		expected_class.erase(0,12);
		std:string img_name = expected_class;
		expected_class.erase(expected_class.find_last_of('.'));
		expected_class.erase(std::remove_if(std::begin(expected_class), std::end(expected_class),[](char ch) { return std::isdigit(ch); }), expected_class.end());

		imshow(img_name, images[i]);

		vector<int> compression_params;
		compression_params.push_back( CV_IMWRITE_JPEG_QUALITY );
		compression_params.push_back( 100 );
		std::string img_path = "./test_results/" + img_name;
		imwrite(img_path,cur_frame, compression_params);

		waitKey(0);
	}

	sds_free(packedImages);
	sds_free(packedOut);
	return 0;
}





/*
	load all the jpg images in ./test_images for testing
	output result to test_results

*/
int classify_frames_load(int win_height, int win_width)
{
	//initialize variables
	cv::Mat reduced_sized_frame(32, 32, CV_8UC3);
	cv::Mat cur_frame;
	Mat bnn_input = Mat(win_width, win_height, CV_8UC3);
	float_t scale_min = -1.0;
    float_t scale_max = 1.0;
	unsigned int frame_num = 0;	
	int number_class = 10;
	int output = 0;
	


	myfile.open ("result_load.csv",std::ios_base::app);
	printf("Hello BNN\n");

	deinit();
	load_parameters(BNN_PARAMS.c_str()); 
	printf("Done loading BNN\n");
	
	// Initialize the network 
	FoldedMVInit("cnv-pynq");
	network<mse, adagrad> nn;
	makeNetwork(nn);

	// Get a list of all the output classes
	vector<string> classes;
	ifstream file((USER_DIR + "params/cifar10/classes.txt").c_str());
	cout << "Opening parameters at: " << (USER_DIR + "params/cifar10/classes.txt") << endl;
	string str;
	if (file.is_open())
	{
		cout << "Classes: [";
		while (getline(file, str))
		{
			cout << str << ", "; 
			classes.push_back(str);
		}
		cout << "]" << endl;
		
		file.close();
	}
	else
	{
		cout << "Failed to open classes.txt" << endl;
	}

	//Loading png images in test_images files
	vector<cv::String> fn;
	glob("test_images/*.jpg", fn, false);

	vector<Mat> images;
	size_t count = fn.size(); 
	for (size_t i=0; i<count; i++)
	{
		cout<<"loading images "<< i <<endl;
		images.push_back(imread(fn[i]));
		//std::string window_name = "Display" + std::to_string(i); //to display the image
		//imshow(window_name, images[i]);
		//waitKey(0);
	}

	// # of ExtMemWords per input
	const unsigned int psi = 384; //paddedSize(imgs.size()*inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
	// # of ExtMemWords per output
	const unsigned int pso = 16; //paddedSize(64*outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
	if(INPUT_BUF_ENTRIES < psi)
	throw "Not enough space in accelBufIn";
	if(OUTPUT_BUF_ENTRIES < pso)
	throw "Not enough space in accelBufOut";
	// allocate host-side buffers for packed input and outputs
	//ExtMemWord * packedImages = new ExtMemWord[(count * psi)];
	ExtMemWord * packedImages = (ExtMemWord *)sds_alloc((count * psi)*sizeof(ExtMemWord));
	//ExtMemWord * packedOut = new ExtMemWord[(count * pso)];
	ExtMemWord * packedOut = (ExtMemWord *)sds_alloc((count * pso)*sizeof(ExtMemWord));


	for (size_t i=0; i<count; i++){
		cur_frame = images[i];
		int img_height = cur_frame.size().height;
		int img_width = cur_frame.size().width;
		//Mat bnn_input = Mat(cv::Size(win_width, win_height));
		std::vector<uint8_t> bgr;

		if(win_height == 0)
		{
			cv::resize(cur_frame, reduced_sized_frame, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC );	
			flatten_mat(reduced_sized_frame, bgr);			
			vec_t img;
			std::transform(bgr.begin(), bgr.end(), std::back_inserter(img),[=](unsigned char c) { return scale_min + (scale_max - scale_min) * c / 255; });
			quantiseAndPack<8, 1>(img, &packedImages[0], psi);										
		} else {	
			// Take only part of the frame from the original frame (center)
			Rect R(Point((img_width/2)-(win_width/2), (img_height/2)-(win_height/2)), Point((img_width/2)+(win_width/2), (img_height/2)+(win_height/2)));

			cv::resize(cur_frame(R), bnn_input, cv::Size(win_width, win_height), 0, 0, cv::INTER_CUBIC );
			cv::resize(bnn_input, reduced_sized_frame, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC );			
			flatten_mat(reduced_sized_frame, bgr);
			vec_t img;
			std::transform(bgr.begin(), bgr.end(), std::back_inserter(img),[=](unsigned char c) { return scale_min + (scale_max - scale_min) * c / 255; });
			quantiseAndPack<8, 1>(img, &packedImages[0], psi);															
		}

		// Call the hardware function
		kernelbnn((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, false, 0, 0, 0, 0, count,psi,pso,1,0);
		if (frame_num != 1)
		{
			kernelbnn((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, false, 0, 0, 0, 0, count,psi,pso,0,1);
		}
		// Extract the output of BNN and classify result
		std::vector<unsigned int> class_result;
		tiny_cnn::vec_t outTest(number_class, 0);
		copyFromLowPrecBuffer<unsigned short>(&packedOut[0], outTest);
		for(unsigned int j = 0; j < number_class; j++) {			
			class_result.push_back(outTest[j]);
		}		
		output = distance(class_result.begin(),max_element(class_result.begin(), class_result.end()));

		cout<< "classification result: " << classes[output] << endl;


		std::string expected_class = fn[i];
		expected_class.erase(0,12);
		std:string img_name = expected_class;
		expected_class.erase(expected_class.find_last_of('.'));
		expected_class.erase(std::remove_if(std::begin(expected_class), std::end(expected_class),[](char ch) { return std::isdigit(ch); }), expected_class.end());

		if (expected_class == classes[output])
		{
			cout<<"correctly identified "<< expected_class <<endl;
		} else{
			cout<<"cannot identify it is a " << expected_class <<endl;
		}

		putText(cur_frame, classes[output], Point((img_width/2)-(win_width/2) + 10, (img_height/2)-(win_height/2)+10), FONT_HERSHEY_PLAIN, 1 , Scalar(0, 255, 0));
		rectangle(cur_frame, Point((img_width/2)-(win_width/2), (img_height/2)-(win_height/2)), Point((img_width/2)+(win_width/2), (img_height/2)+(win_height/2)), Scalar(0, 0, 255)); // draw a 32x32 box at the centre
		imshow(img_name, cur_frame);

		vector<int> compression_params;
		compression_params.push_back( CV_IMWRITE_JPEG_QUALITY );
		compression_params.push_back( 100 );
		std::string img_path = "./test_results/" + img_name;
		imwrite(img_path,cur_frame, compression_params);

		waitKey(0);
	}

	sds_free(packedImages);
	sds_free(packedOut);
	return 0;
}

// This function has BNN, software functions: capturing + pre-processing functions all piplined
int classify_frames_cam_sw (int frames, int win_height, int win_width, int win_step, int win_length, int argv7)
{

	//variable initialization
	int number_class = 10;
	int certainty_spread = 10;
	int	output = 0;
	unsigned int frame_num = 0;	
	const unsigned int count = 1;
    float_t scale_min = -1.0;
    float_t scale_max = 1.0;
	std::vector<uint8_t> bgr;
	vector<float> certainty;
	cv::Mat reduced_sized_frame(32, 32, CV_8UC3);
	int expected_class_num = argv7;
	unsigned int identified = 0 , identified_adj = 0;

	// # of ExtMemWords per input
	const unsigned int psi = 384; //paddedSize(imgs.size()*inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
	// # of ExtMemWords per output
	const unsigned int pso = 16; //paddedSize(64*outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
	if(INPUT_BUF_ENTRIES < psi)
	throw "Not enough space in accelBufIn";
	if(OUTPUT_BUF_ENTRIES < pso)
	throw "Not enough space in accelBufOut";
	// allocate host-side buffers for packed input and outputs
	//ExtMemWord * packedImages = new ExtMemWord[(count * psi)];
	ExtMemWord * packedImages = (ExtMemWord *)sds_alloc((count * psi)*sizeof(ExtMemWord));
	//ExtMemWord * packedOut = new ExtMemWord[(count * pso)];
	ExtMemWord * packedOut = (ExtMemWord *)sds_alloc((count * pso)*sizeof(ExtMemWord));

	std::vector<float> weights;
	weights.resize(win_length);
	std::vector<std::vector<float> > results_history; // All its previous classifications
	int step_counts = 0;
	int past_output = 0;
	float lambda = 0.2;
	// Pre-populate history weights with exponential decays
	for(int i = 0; i < win_length; i++)
	{
		weights[i] = expDecay(lambda, i);
	}

	Mat cur_frame, cap_frame;
	std::string window_mode;


	myfile.open ("result_SW.csv",std::ios_base::app);
	printf("Hello BNN\n");

	cout << "  Number of processors available = " << omp_get_num_procs ( ) << "\n";
	cout << "  Number of threads =              " << omp_get_max_threads ( ) << "\n";
	
	deinit();
	load_parameters(BNN_PARAMS.c_str()); 
	printf("Done loading BNN\n");
	
	// Initialize the network 
	FoldedMVInit("cnv-pynq");
	network<mse, adagrad> nn;
	makeNetwork(nn);

	// Get a list of all the output classes
	vector<std::string> classes;
	ifstream file((USER_DIR + "params/cifar10/classes.txt").c_str());
	cout << "Opening parameters at: " << (USER_DIR + "params/cifar10/classes.txt") << endl;
	string str;
	if (file.is_open())
	{
		cout << "Classes: [";
		while (getline(file, str))
		{
			cout << str << ", "; 
			classes.push_back(str);
		}
		cout << "]" << endl;
		
		file.close();
	}
	else
	{
		cout << "Failed to open classes.txt" << endl;
	}



	// Open defult camera
	VideoCapture cap(0 + CV_CAP_V4L2);
	if(!cap.open(0))
	{
	cout << "cannot open camera" << endl;
	return 0;
	}

	cap.set(CV_CAP_PROP_FRAME_WIDTH,frame_width);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT,frame_height);
	std::cout << "\nCamera resolution = " << cap.get(CV_CAP_PROP_FRAME_WIDTH) << "x" << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << std::endl;
	
	
	

	if(win_height == 0)
	{
		window_mode = "Full window";
	}
	else 
	{
		window_mode = "Sub Window";		
	}

	myfile << "\nFrame No., Camera Time(us), Post Processing Time(us), Total Time(us), Output , Adjusted Output, , Camera Frame Rate, Classification Rate \n";
	
	while(true)
	{
		cout << "\nStart while loop (HW and SW multithreading and piplined):" << endl;
		auto t1 = chrono::high_resolution_clock::now(); //time statistics

		if (frame_num == 0)
		{
			cap >> cur_frame;
		}

		#pragma omp parallel sections
		{
			#pragma omp section
			{
				cap >> cap_frame;
				cur_frame = cap_frame;
			}
			
			#pragma omp section
			{
				if(win_height == 0)
				{
					cv::resize(cur_frame, reduced_sized_frame, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC );	
					flatten_mat(reduced_sized_frame, bgr);			
					vec_t img;
					std::transform(bgr.begin(), bgr.end(), std::back_inserter(img),
						[=](unsigned char c) { return scale_min + (scale_max - scale_min) * c / 255; });
					quantiseAndPack<8, 1>(img, &packedImages[0], psi);										
				}
				else
				{	// Take only 128*128 frame size from the original frame
					Rect R(Point((frame_width/2)-(win_width/2), (frame_height/2)-(win_height/2)), Point((frame_width/2)+(win_width/2), (frame_height/2)+(win_height/2)));
					Mat bnn_input;
					bnn_input = cur_frame(R); //Extract center frame as ROI
				
					cv::resize(bnn_input, reduced_sized_frame, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC );	
					flatten_mat(reduced_sized_frame, bgr);			
					vec_t img;
					std::transform(bgr.begin(), bgr.end(), std::back_inserter(img),
						[=](unsigned char c) { return scale_min + (scale_max - scale_min) * c / 255; });
					quantiseAndPack<8, 1>(img, &packedImages[0], psi);															
				}
			}			
		}

		frame_num++;

		auto t2 = chrono::high_resolution_clock::now();	//time statistics		
		auto camera_processing_time = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count(); //time statistics	
		
		// Call the hardware function
		kernelbnn((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, false, 0, 0, 0, 0, count,psi,pso,1,0);
		if (frame_num != 1)
		{
			kernelbnn((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, false, 0, 0, 0, 0, count,psi,pso,0,1);
		}
		// Extract the output of BNN and classify result
		std::vector<unsigned int> class_result;
		tiny_cnn::vec_t outTest(number_class, 0);
		copyFromLowPrecBuffer<unsigned short>(&packedOut[0], outTest);
		for(unsigned int j = 0; j < number_class; j++) {			
			class_result.push_back(outTest[j]);
		}		
		output = distance(class_result.begin(),max_element(class_result.begin(), class_result.end()));	


		// Data Post Processing
		// update result_history
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

		auto t3 = chrono::high_resolution_clock::now();
		auto post_processing_time = chrono::duration_cast<chrono::microseconds>( t3 - t2 ).count();
		auto total_time = chrono::duration_cast<chrono::microseconds>( t3 - t1 ).count();
	  	
		cout << "Output = " << classes[output] << endl;	
		cout << "Adjusted Output = " << classes[adjusted_output] << endl;	
		
		if (expected_class_num == output){
			identified ++;
		}
		if (expected_class_num == adjusted_output){
			identified_adj++;
		}

		std::cout << "Frame number is: " << frame_num << std::endl;	
		std::cout << "Camera processing time(us) is: " << (float)camera_processing_time << std::endl;
		std::cout << "Data post processing time(us) is: " << (float)post_processing_time << std::endl;
		std::cout << "Total time(us) is: " << (float)total_time << std::endl;
		
		putText(cur_frame, classes[output], Point(15, 55), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));		
		putText(cur_frame, classes[adjusted_output], Point(15, 75), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));						
		putText(cur_frame, window_mode, Point(15, 115), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));				
		if (win_height != 0)
		{			
			rectangle(cur_frame, Point((frame_width/2)-(win_width/2), (frame_height/2)-(win_height/2)), Point((frame_width/2)+(win_width/2), (frame_height/2)+(win_height/2)), Scalar(0, 0, 255)); // draw a 32x32 box at the centre
		}

		float camera_frame_rate = 1/((float)camera_processing_time/1000000);
		float classification_rate = 1/((float)total_time/1000000);
        myfile << frame_num << "," << camera_processing_time << "," << post_processing_time << "," << total_time << "," << classes[output] << "," << classes[adjusted_output]<< "," << "" << "," << camera_frame_rate << "," << classification_rate <<"\n";

		imshow("Original", cur_frame);
		char ESC = waitKey(1);
		if (frame_num > frames) 
        {
            cout << "Number of frames done: " << frame_num << endl;
            break;
        }		
		if (ESC == 27) 
        {
            cout << "ESC key is pressed by user" << endl;
            break;
        }		
	}
	
	float accuracy = 100.0*((float)identified/(float)frame_num);
	float accuracy_adj = 100.0*((float)identified_adj/(float)frame_num);
	myfile << "\n Accuracy," << accuracy << "," << classes[expected_class_num];
	myfile << "\n Adjusted Accuracy," << accuracy_adj << "," << classes[expected_class_num];
	myfile << "\n Number of frames classified," << frame_num;
	myfile << "\n Number of frames identified," << identified;
	myfile << "\n Frame Size," << window_mode << "," << win_width <<"x"<< win_height;
	
	cap.release();
	myfile.close();	
	sds_free(packedImages);
	sds_free(packedOut);
	return 1;
}

