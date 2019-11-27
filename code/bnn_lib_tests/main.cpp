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


using namespace std;
using namespace tiny_cnn;
using namespace tiny_cnn::activation;
using namespace cv;

#define FRAME_WIDTH 320		//176	//320	//640
#define FRAME_HEIGHT 240		//144	//240	//480

unsigned int WINDOW_WIDTH;
unsigned int WINDOW_HEIGHT;
unsigned int g_win_step;
unsigned int g_win_length;
float lambda;

const std::string USER_DIR = "/home/xilinx/jose_bnn/bnn_lib_tests/";
const std::string BNN_PARAMS = USER_DIR + "params/cifar10/";
unsigned int ok, failed, ok_adjusted;
int expected_class_num;
int runFrames, smallWindow;

ofstream myfile;

int calssifyCameraFrames1();
int calssifyCameraFrames2();


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

	@para arg_count:
	@para arg_results_history:
	@para arg_weights
	@return output
*/
int output_filter(int arg_count, std::vector<std::vector<float>> arg_results_history, std::vector<float> arg_weights){  	

	cout << "output filer starts here" << endl;
	if (arg_count < g_win_step){
		std::vector<float> past_result = arg_results_history.rbegin()[arg_count];
		return distance(past_result.begin(), max_element(past_result.begin(), past_result.end()));
	}else if ( arg_count = g_win_step){

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

int main(int argc, char** argv)
{
	for(int i = 0; i < argc; i++)
		cout << "argv[" << i << "]" << " = " << argv[i] << endl;	
	
	if(argc == 4)
	{
		if (strcmp(argv[1], "hw") == 0)
		{
			expected_class_num = atoi(argv[2]);
			runFrames = atoi(argv[3]);
			calssifyCameraFrames1();
		}
		else
		{
			helpMessage(argc, argv);
			return -1;
		}
	}
	
	else if(argc == 7)
	{
		//	./BNN sw 1 1000 128 5 12
		if (strcmp(argv[1], "sw") == 0)
		{
			expected_class_num = atoi(argv[2]);
			runFrames = atoi(argv[3]);
			WINDOW_HEIGHT = atoi(argv[4]);
			WINDOW_WIDTH = atoi(argv[4]);
			g_win_step = atoi(argv[5]);//5
			g_win_length = atoi(argv[6]);//12
			if (WINDOW_HEIGHT == 0)
				smallWindow = 0; 
			else 
				smallWindow = 1; 
			calssifyCameraFrames2();
		}
		else
		{
			helpMessage(argc, argv);
			return -1;
		}
		
	}
	else
	{
		cout << "argc = " << argc << endl;
		helpMessage(argc, argv);
		return -1;
	}
	
}


// This function has BNN and the software function piplined
int calssifyCameraFrames1()
{
	myfile.open ("result_HW.csv",std::ios_base::app);
	printf("Hello BNN\n");

	//cout << "  Number of processors available = " << omp_get_num_procs ( ) << "\n";
	//cout << "  Number of threads =              " << omp_get_max_threads ( ) << "\n";
	
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

	// Open defult camera
	VideoCapture cap(0);
    if(!cap.open(0))
    {
       cout << "cannot open camera" << endl;
       return 0;
    }

	cap.set(CV_CAP_PROP_FRAME_WIDTH,FRAME_WIDTH);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT);
	
	std::cout << "\nCamera resolution = " << cap.get(CV_CAP_PROP_FRAME_WIDTH) << "x" << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << std::endl;

	int number_class = 10;
	int certainty_spread = 10;
	int	output = 0;
	unsigned int frameNum = 0;	
	const unsigned int count = 1;
    float_t scale_min = -1.0;
    float_t scale_max = 1.0;
	std::vector<uint8_t> bgr;
	cv::Mat reducedSizedFrame(32, 32, CV_8UC3);

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

	myfile << "\nframeNum, cap_time, process_time, BNN_time, total_time, classes[output]\n";
	
	while(true)
	{	
		cout << "\nStart while loop (HW and SW piplined):" << endl;
		auto t1 = chrono::high_resolution_clock::now();	
		clock_t beginFrame = clock();

		Mat curFrame;
		cap >> curFrame;
        if( curFrame.empty() ) break; // end of video stream
		frameNum++;	
		auto t2 = chrono::high_resolution_clock::now();		
		// Pre-process frames
		cv::resize(curFrame, reducedSizedFrame, cv::Size(32, 32), 0, 0, cv::INTER_LINEAR );	
		flatten_mat(reducedSizedFrame, bgr);
		// Scale image pixel values to float between -1 and 1
		vec_t img;
		std::transform(bgr.begin(), bgr.end(), std::back_inserter(img),
			[=](unsigned char c) { return scale_min + (scale_max - scale_min) * c / 255; });
		// Convert float to binary format by quantizing and suing SDSoC data format
		quantiseAndPack<8, 1>(img, &packedImages[0], psi);
		auto t3 = chrono::high_resolution_clock::now();	
		auto capPreprocess_time = chrono::duration_cast<chrono::microseconds>( t3 - t1 ).count();
/*
		while(capPreprocess_time < 8000) // fix program to 125 FPS
		{
			t3 = chrono::high_resolution_clock::now();
			capPreprocess_time = chrono::duration_cast<chrono::microseconds>( t3 - t1 ).count();
		}*/
		
		/*
		if (frameNum != 1)
		{
		kernelbnn((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, false, 0, 0, 0, 0, count,psi,pso,0,1);	
		}*/
		// Call the hardware function
		kernelbnn((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, false, 0, 0, 0, 0, count,psi,pso,1,0);
		if (frameNum != 1)
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
		//float certainty = calculate_certainty(class_result, certainty_spread);	
		output = distance(class_result.begin(),max_element(class_result.begin(), class_result.end()));	
		auto t4 = chrono::high_resolution_clock::now();		

		auto total_time = chrono::duration_cast<chrono::microseconds>( t4 - t1 ).count();

		
		cout << "Output = " << classes[output] << endl;	
		if ( expected_class_num == output)
			ok++;
		//cout << "certainty = " << certainty << endl;

		std::cout << "frame number is: " << frameNum << std::endl;		
		auto cap_time = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
		cout << "cap_time :" << cap_time << " microseconds, " << 1000000/(float)cap_time << " FPS" << endl;
		auto process_time = chrono::duration_cast<chrono::microseconds>( t3 - t2 ).count();
		cout << "process_time :" << process_time << " microseconds, " << 1000000/(float)process_time << " FPS" << endl;
		auto BNN_time = chrono::duration_cast<chrono::microseconds>( t4 - t3 ).count();
		cout << "BNN_time :" << BNN_time << " microseconds, " << 1000000/(float)BNN_time << " FPS" << endl;
		//auto total_time = chrono::duration_cast<chrono::microseconds>( t4 - t1 ).count();
		cout << "total_time :" << total_time << " microseconds, " << 1000000/(float)total_time << " FPS" << endl;

        myfile << frameNum << "," << cap_time << "," << process_time << "," << BNN_time << "," << total_time << "," << classes[output] <<"\n";

		putText(curFrame, classes[output], Point(55, 55), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));		
		//putText(curFrame, to_string(certainty), Point(55, 75), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));				
		imshow("Original", curFrame);
		char ESC = waitKey(1);
		
		if (frameNum > runFrames) 
        {
            cout << "Number of frames done: " << frameNum << endl;
            break;
        }		
		if (ESC == 27) 
        {
            cout << "ESC key is pressed by user" << endl;
            break;
        }		
	}
	float Accuracy = 100.0*((float)ok/(float)frameNum);
	myfile << "\n Accuracy," << Accuracy << "," << classes[expected_class_num];
	myfile << "\n Number of program frames classified," << frameNum;
	
	cap.release();
	myfile.close();	
	sds_free(packedImages);
	sds_free(packedOut);
}



// This function has BNN, software functions: capturing + pre-processing functions all piplined
int calssifyCameraFrames2()
{
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

	// Open defult camera
	VideoCapture cap(0 + CV_CAP_V4L2);
    if(!cap.open(0))
    {
       cout << "cannot open camera" << endl;
       return 0;
    }

	cap.set(CV_CAP_PROP_FRAME_WIDTH,FRAME_WIDTH);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT);
	
	std::cout << "\nCamera resolution = " << cap.get(CV_CAP_PROP_FRAME_WIDTH) << "x" << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << std::endl;

	int number_class = 10;
	int certainty_spread = 10;
	int	output = 0;
	unsigned int frameNum = 0;	
	const unsigned int count = 1;
    float_t scale_min = -1.0;
    float_t scale_max = 1.0;
	std::vector<uint8_t> bgr;
	vector<float> certainty;
	cv::Mat reducedSizedFrame(32, 32, CV_8UC3);
	unsigned int window_step_count = 0;

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
	weights.resize(g_win_length);
	std::vector<std::vector<float> > results_history; // All its previous classifications
	int step_counts = 0;
	float lambda = 0.2;
	// Pre-populate history weights with exponential decays
	for(int i = 0; i < g_win_length; i++)
	{
		weights[i] = expDecay(lambda, i);
	}


	Mat curFrame, capFrame;
	string windowMode;
	if(smallWindow == 0)
	{
		windowMode = "Full window";
	}
	else 
	{
		windowMode = "Sub Window";		
	}
	myfile << "\nframeNum, capPreprocess_time, BNN_time, total_time, classes[output]\n";
	
	while(true)
	{
		cout << "\nStart while loop (HW and SW multithreading and piplined):" << endl;
		auto t1 = chrono::high_resolution_clock::now();	
		// Capture the first frame to input to second processor
		if (frameNum == 0)
		{
			cap >> curFrame;
		}

		#pragma omp parallel sections
		{
			#pragma omp section
			{
				cap >> capFrame;
				curFrame = capFrame;
				//if( curFrame.empty() ) break; // end of video stream
				//cout << "capture frame P1" << endl;
			}
			
			#pragma omp section
			{
				if(smallWindow == 0)
				{
					cv::resize(curFrame, reducedSizedFrame, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC );	
					flatten_mat(reducedSizedFrame, bgr);			
					vec_t img;
					std::transform(bgr.begin(), bgr.end(), std::back_inserter(img),
						[=](unsigned char c) { return scale_min + (scale_max - scale_min) * c / 255; });
					quantiseAndPack<8, 1>(img, &packedImages[0], psi);		
					//cout << "full frame" << endl;									
				}
				else
				{	// Take only 128*128 frame size from the original frame
					Rect R(Point((FRAME_WIDTH/2)-(WINDOW_WIDTH/2), (FRAME_HEIGHT/2)-(WINDOW_HEIGHT/2)), Point((FRAME_WIDTH/2)+(WINDOW_WIDTH/2), (FRAME_HEIGHT/2)+(WINDOW_HEIGHT/2)));
					Mat bnnInput;
					bnnInput = curFrame(R); //Extract ROI
				
					cv::resize(bnnInput, reducedSizedFrame, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC );	
					flatten_mat(reducedSizedFrame, bgr);			
					vec_t img;
					std::transform(bgr.begin(), bgr.end(), std::back_inserter(img),
						[=](unsigned char c) { return scale_min + (scale_max - scale_min) * c / 255; });
					quantiseAndPack<8, 1>(img, &packedImages[0], psi);							
					//cout << "reduced frame" << endl;									
				}
			}			
		}

		frameNum++;			
		auto t2 = chrono::high_resolution_clock::now();			
		auto capPreprocess_time = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
		/*
		while(capPreprocess_time < 8000)
		{
			t2 = chrono::high_resolution_clock::now();
			capPreprocess_time = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
		}*/		
		
		// Call the hardware function
		kernelbnn((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, false, 0, 0, 0, 0, count,psi,pso,1,0);
		if (frameNum != 1)
		{
			kernelbnn((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, false, 0, 0, 0, 0, count,psi,pso,0,1);
		}
		//usleep(500);
		// Extract the output of BNN and classify result
		std::vector<unsigned int> class_result;
		tiny_cnn::vec_t outTest(number_class, 0);
		copyFromLowPrecBuffer<unsigned short>(&packedOut[0], outTest);
		for(unsigned int j = 0; j < number_class; j++) {			
			class_result.push_back(outTest[j]);
		}		
		output = distance(class_result.begin(),max_element(class_result.begin(), class_result.end()));	


		// Data Post Processing
		//update result_history
		results_history.insert(results_history.begin(), calculate_certainty(class_result));
		if (results_history.size() > g_win_length)
		{
			results_history.pop_back();
		}

		int adjusted_output = 0;
		adjusted_output = output_filter(step_counts, results_history, weights);

		// if (step_counts < g_win_step) {
		// 	step_counts++;
		// } else {
		// 	step_counts = 0;
		// }
	  	
		cout << "Output = " << classes[output] << endl;	
		cout << "Adjusted Output = " << classes[adjusted_output] << endl;	
		
		if ( expected_class_num == output)
			ok++;
		if ( expected_class_num == adjusted_output)
			ok_adjusted++;

		std::cout << "frame number is: " << frameNum << std::endl;		
		// //auto capPreprocess_time = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
		// cout << "capPreprocess_time :" << capPreprocess_time << " microseconds, " << 1000000/(float)capPreprocess_time << " FPS" << endl;
		// //auto BNN_time = chrono::duration_cast<chrono::microseconds>( t3 - t2 ).count();
		// cout << "BNN_time :" << BNN_time << " microseconds, " << 1000000/(float)BNN_time << " FPS" << endl;
		// auto total_time = chrono::duration_cast<chrono::microseconds>( t3 - t1 ).count();
		// cout << "total_time :" << total_time << " microseconds, " << 1000000/(float)total_time << " FPS" << endl;

		putText(curFrame, classes[output], Point(15, 55), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));		
		putText(curFrame, classes[adjusted_output], Point(15, 75), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));		
		//putText(curFrame, to_string(max_result), Point(15, 95), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));				
		putText(curFrame, windowMode, Point(15, 115), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));				
		if (smallWindow != 0)
		{			
			rectangle(curFrame, Point((FRAME_WIDTH/2)-(WINDOW_WIDTH/2), (FRAME_HEIGHT/2)-(WINDOW_HEIGHT/2)), Point((FRAME_WIDTH/2)+(WINDOW_WIDTH/2), (FRAME_HEIGHT/2)+(WINDOW_HEIGHT/2)), Scalar(0, 0, 255)); // draw a 32x32 box at the centre
		}
        //myfile << frameNum << "," << capPreprocess_time << "," << BNN_time << "," << total_time << "," << classes[output] <<"\n";

		imshow("Original", curFrame);
		char ESC = waitKey(1);
		if (frameNum > runFrames) 
        {
            cout << "Number of frames done: " << frameNum << endl;
            break;
        }		
		if (ESC == 27) 
        {
            cout << "ESC key is pressed by user" << endl;
            break;
        }		
	}
	float Accuracy = 100.0*((float)ok/(float)frameNum);
	float Accuracy_adj = 100.0*((float)ok_adjusted/(float)frameNum);
	myfile << "\n Accuracy," << Accuracy << "," << classes[expected_class_num];
	myfile << "\n Adjusted Accuracy," << Accuracy_adj << "," << classes[expected_class_num];
	myfile << "\n Number of frames classified," << frameNum;
	myfile << "\n Frame Size," << windowMode << "," << WINDOW_WIDTH <<"x"<<WINDOW_WIDTH;
	
	cap.release();
	myfile.close();	
	sds_free(packedImages);
	sds_free(packedOut);
}