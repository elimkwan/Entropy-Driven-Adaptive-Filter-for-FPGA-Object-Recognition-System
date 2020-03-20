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

	File use for automize Window Filter Testing
	SRC: Load images from ./experiments/dataset + "foldernum"
	DEST: ./experiement/result
	


	Command avaliable:
	./BNN

	(change config at ~ line 366-400)

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
//const std::string TEST_DIR = "/home/xilinx/jose_bnn/bnn_lib_tests/experiments/";

ofstream myfile;
ofstream fs;

//main functions
int wrapper(unsigned int no_of_frame, string uncertainty_config, bool dropf_config, bool win_config, string roi_config, bool dynclk);
//int classify_frames(unsigned int no_of_frame, string uncertainty_config, bool dropf_config, bool win_config, string roi_config, bool dynclk);
void config_clock(int desired_frequency);

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

// double clockToMilliseconds(clock_t ticks){
//     // units/(units/time) => time (seconds) * 1000 = milliseconds
//     return (ticks/(double)CLOCKS_PER_SEC)*1000.0;
// }

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

int main(int argc, char** argv)
{
	int no_of_frame = 1000;
	std::string uncertainty_config = "en";
	std::string roi_config = "full-roi";

	bool dropf_bool = false;
	bool win_bool = false;
	bool dynclk_bool = false;
	bool base_bool = false;
	int wstep = 1;
	int wlength = 10;

	config_clock(0);

	wrapper(no_of_frame, uncertainty_config, dropf_bool, win_bool, roi_config, dynclk_bool);

	return 1;
}

void config_clock(int desired_frequency){
	//configuration of PL clocks
	cout << "Starting PL clock configuration: " << endl;

	int memfd;
	void *mapped_base, *mapped_dev_base;
	off_t dev_base = HW_ADDR_GPIO; //GPIO hardware


	memfd = open("/dev/mem", O_RDWR | O_SYNC);
	if (memfd == -1) {
		printf("Can't open /dev/mem.\n");
		exit(0);
	}
	printf("/dev/mem opened for gpio.\n");

	// Map one page of memory into user space such that the device is in that page, but it may not
	// be at the start of the page.
	mapped_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, dev_base & ~MAP_MASK);
	if (mapped_base == (void *) -1) {
		printf("Can't map the memory to user space.\n");
		exit(0);
	}
	printf("GPIO mapped at address %p.\n", mapped_base);


	// get the address of the device in user space which will be an offset from the base
	// that was mapped as memory is mapped at the start of a page

	mapped_dev_base = mapped_base + (dev_base & MAP_MASK);

	int* pl_clk = (int*)mapped_dev_base;

	cout << "Current PL clock configuration: " << hex << *pl_clk << endl;

	if (desired_frequency == 100){
		*pl_clk = 0x00100600; //166 MHz
	} else if (desired_frequency == 50){
		//*pl_clk = 0x00A00200;
		*pl_clk = 0x00A00400; //25MHz
	} else if (desired_frequency == 0){
		*pl_clk = 0x00A00100; //100MHz
	}

	cout << "New PL clock configuration: " << hex << *pl_clk << endl;
	cout << dec;

}

int wrapper(unsigned int no_of_frame, string uncertainty_config, bool dropf_config, bool win_config, string roi_config, bool dynclk){


	const unsigned int count = 1;
	float_t scale_min = -1.0;
	float_t scale_max = 1.0;

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
	// cout<<"debug memory1: " << &packedImages << endl;
	// cout<<"debug memory2: " << &packedOut << endl;


	fs.open ("./experiments/result/result-overview.csv",std::ios_base::app);
	fs << "\nSetting, Window Step, Window Length, Accuracy, Adjusted Accuracy, Performance Gain, Avg. Classification Rate, Avg. Processing Rate \n" ;

	vector<vector<int>> ws_wl = {{1, 24}, {4, 24},{8,24}, {23,24}, {4,12}, {8,12} };
	//vector<vector<int>> ws_wl = {{4, 24}};
	vector<int> dataset_list = {2,7};
	bool base = false;
	int wstep, wlength, folder_num;

	for (int i = 0; i < dataset_list.size(); i++){

		folder_num = dataset_list[i];

		if ( folder_num  == 3){
			no_of_frame = 500;
		} else{
			no_of_frame = 1000;
		}

		//loading dataset
		vector<Mat> frames;
		vector<cv::String> fn;
		string src_dir =  TEST_DIR + "dataset" + to_string(folder_num) + "/*.png";
		glob(src_dir, fn, false);
		//print_vector(fn);
		size_t count_fn = fn.size();
		for (size_t i=0; i<count_fn; i++)
		{
			cout<<"loading images "<< i <<endl;
			frames.push_back(imread(fn[i]));
		}
		cout<<"loaded all images " <<endl;

		for (int j = 0; j < ws_wl.size(); j ++){
			wstep = ws_wl[j][0];
			wlength = ws_wl[j][1];

			//----------------------------------------------------------------------------------------------------------------
			//----------------------------------------------------------------------------------------------------------------
			//----------------------------------------------------------------------------------------------------------------

			//Initialize variables
			cv::Mat reduced_sized_frame(32, 32, CV_8UC3);
			cv::Mat cur_frame, src, reduced_roi_frame;
			//Mat bnn_input = Mat(frame_size, frame_size, CV_8UC3);
			unsigned int number_class = 10;
			unsigned int output = 0;
			vector<string> classes = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
			unsigned int size, frame_num = 0;
			tiny_cnn::vec_t outTest(number_class, 0);
			std::vector<uint8_t> bgr;
			std::vector<std::vector<float> > results_history; //for storing the classification result of previous frame
			float identified = 0.0 , identified_adj = 0.0, total_time = 0.0;

			
			string foldername = "setting" + to_string(folder_num) + "-" + to_string(wstep) + "-" + to_string(wlength);
			if (base) {
				foldername = foldername + "-base";
			}	
			std::cout << "-------------------------------------------------"<< endl;
			std::cout << foldername << endl;
			std::cout << "-------------------------------------------------"<< endl;

			string result_dir = "./experiments/result/" + foldername + ".csv"; 
			myfile.open (result_dir,std::ios_base::app);
			myfile << foldername << "\n";
			myfile << "\nFrame No., classification rate(fps),  processing rate(fps) , Output , Adjusted Output, Simulate Cap Time(us), preprocess_time(us), bnn_time(us), window_filter_time(us), uncertainty_time(us), en/var/a , ma, sd, state, mode\n";

			cur_frame = frames[0];

			Roi_filter r_filter(frame_width,frame_height);
			r_filter.init_enhanced_roi(cur_frame);

			//output filter with windowing techniques
			//Win_filter w_filter(0.2f, 8, 12);
			Win_filter w_filter(0.2f, wstep, wlength);
			w_filter.init_weights(0.2f);
			//cout << "size of weight:" << w_filter.wweights.size() << endl;

			Uncertainty u_filter(5);
			//Uncertainty var_filter(5);//testing various uncertainty scheme

			int drop_frame_mode = 0;
			int frames_dropped = 0;
			unsigned int adjusted_output = 0;
			cv::Mat display_frame;
			int pastclk = 100;
			float acc_time = 0;
			int processed_frames = 0;
			string display_output = "";
			Rect roi(Point(0,0), Point(frame_width, frame_height));
			Rect display_roi(Point(0,0), Point(frame_width, frame_height));
			float cap_time, preprocessing_time = 0.0;
			vector<float> time_arr;
			float processing_fps = 0;
			bool not_dropping_frame = true;
			float bnn_time, uncertainty_time, wfilter_time, en_time, var_time;
			vector<double> u(5, 0.0);
			vector<float> process_rate_arr;


			while(frame_num < count_fn){

				bool c30 = ((frame_num+1)%5 == 0);
				auto t0 = chrono::high_resolution_clock::now(); //time statistics

				#pragma omp parallel sections
				{
					#pragma omp section
					{

						cur_frame = frames[frame_num];
						waitKey(5);

						auto t1 = chrono::high_resolution_clock::now();	//time statistics
						cap_time = chrono::duration_cast<chrono::microseconds>( t1 - t0 ).count();
					}

					#pragma omp section
					{


						if (base) {
							not_dropping_frame = (frame_num < 2 || frames_dropped == 4); //to set fps to 30fps manually
						} else {
							not_dropping_frame = ( drop_frame_mode == 0 || drop_frame_mode == 1 || drop_frame_mode == 2 || drop_frame_mode == 3 || (drop_frame_mode == 4 && frames_dropped == 5) || (drop_frame_mode == 5 && frames_dropped == 10));
						}

						if (c30 || frame_num == 0){
							display_frame = cur_frame.clone();
						}

						auto t1 = chrono::high_resolution_clock::now(); //time statistics
						if (roi_config == "eff-roi"){

							cv::resize(cur_frame, reduced_roi_frame, cv::Size(80, 60), 0, 0, cv::INTER_CUBIC);
							if (drop_frame_mode != 1){
								r_filter.init_enhanced_roi(reduced_roi_frame);
							}

							if (drop_frame_mode == 0){

								roi = r_filter.get_full_roi();

							}else if (drop_frame_mode == 1){

								roi = r_filter.enhanced_roi(reduced_roi_frame);

							}else if (drop_frame_mode == 2){

								roi = r_filter.basic_roi(reduced_roi_frame);

							}else{
								roi = r_filter.get_past_roi();
							}

							src = cur_frame(roi);
							cv::resize(src, reduced_sized_frame, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC );
							flatten_mat(reduced_sized_frame, bgr);
							vec_t img;
							std::transform(bgr.begin(), bgr.end(), std::back_inserter(img),[=](unsigned char c) { return scale_min + (scale_max - scale_min) * c / 255; });
							quantiseAndPack<8, 1>(img, &packedImages[0], psi);

						} else if (roi_config == "opt-roi"){

							cv::resize(cur_frame, reduced_roi_frame, cv::Size(80, 60), 0, 0, cv::INTER_CUBIC );

							if (frame_num < 2){
								roi = r_filter.get_full_roi();
								r_filter.init_enhanced_roi(reduced_roi_frame);
							} else {
								roi = r_filter.enhanced_roi(reduced_roi_frame);
							}

							src = cur_frame(roi);
							cv::resize(src, reduced_sized_frame, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC );
							flatten_mat(reduced_sized_frame, bgr);
							vec_t img;
							std::transform(bgr.begin(), bgr.end(), std::back_inserter(img),[=](unsigned char c) { return scale_min + (scale_max - scale_min) * c / 255; });
							quantiseAndPack<8, 1>(img, &packedImages[0], psi);


						} else if (roi_config == "cont-roi") {
							
							cv::resize(cur_frame, reduced_roi_frame, cv::Size(80, 60), 0, 0, cv::INTER_CUBIC );

							if (frame_num < 2){
								roi = r_filter.get_full_roi();
							} else {
								roi = r_filter.basic_roi(reduced_roi_frame);
							}

							src = cur_frame(roi);
							cv::resize(src, reduced_sized_frame, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC );
							flatten_mat(reduced_sized_frame, bgr);
							vec_t img;
							std::transform(bgr.begin(), bgr.end(), std::back_inserter(img),[=](unsigned char c) { return scale_min + (scale_max - scale_min) * c / 255; });
							quantiseAndPack<8, 1>(img, &packedImages[0], psi);


						} else if (roi_config == "full-roi") {

							//use full frame all the time, no roi
							cv::resize(cur_frame, reduced_sized_frame, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC );
							flatten_mat(reduced_sized_frame, bgr);
							vec_t img;
							std::transform(bgr.begin(), bgr.end(), std::back_inserter(img),[=](unsigned char c) { return scale_min + (scale_max - scale_min) * c / 255; });
							quantiseAndPack<8, 1>(img, &packedImages[0], psi);

						} else {

							roi = r_filter.naive_roi(cur_frame, 128);
							src = cur_frame(roi);

							cv::resize(src, reduced_sized_frame, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC );
							flatten_mat(reduced_sized_frame, bgr);
							vec_t img;
							std::transform(bgr.begin(), bgr.end(), std::back_inserter(img),[=](unsigned char c) { return scale_min + (scale_max - scale_min) * c / 255; });
							quantiseAndPack<8, 1>(img, &packedImages[0], psi);
						}

						auto t2 = chrono::high_resolution_clock::now();	//time statistics
						preprocessing_time = chrono::duration_cast<chrono::microseconds>( t2 - t0 ).count();
					}
				}

				if (dynclk){
					if ( (drop_frame_mode == 4 || drop_frame_mode == 5) && pastclk == 100){
						config_clock(50);
						pastclk = 50;
					} else if ((drop_frame_mode == 0 || drop_frame_mode == 1 || drop_frame_mode == 2 || drop_frame_mode == 3) && pastclk == 50){
						config_clock(100);
						pastclk = 100;
					}
				}

				if (!not_dropping_frame && (dropf_config || base)){
					frames_dropped +=1;//drop the frame

				} else {
					//reset frames_dropped
					frames_dropped = 0; 
					
					auto t3 = chrono::high_resolution_clock::now();	//time statistics
					// Call the hardware function
					kernelbnn((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, false, 0, 0, 0, 0, count,psi,pso,1,0);
					if (frame_num != 1)
					{
						kernelbnn((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, false, 0, 0, 0, 0, count,psi,pso,0,1);
					}
					// Extract the output of BNN and classify result
					std::vector<float> class_result;
					copyFromLowPrecBuffer<unsigned short>(&packedOut[0], outTest);
					for(unsigned int j = 0; j < number_class; j++) {			
						class_result.push_back(outTest[j]);
					}

					output = distance(class_result.begin(),max_element(class_result.begin(), class_result.end()));

					auto t4 = chrono::high_resolution_clock::now();	//time statistics
					bnn_time = chrono::duration_cast<chrono::microseconds>( t4 - t3 ).count();

					//Data post-processing:
					//calculate uncertainty
					u = u_filter.cal_uncertainty(class_result,uncertainty_config, output);
					drop_frame_mode = u[4];

					auto t5 = chrono::high_resolution_clock::now();	//time statistics
					uncertainty_time = chrono::duration_cast<chrono::microseconds>( t5 - t4 ).count();

					//use window
					w_filter.update_memory(class_result);
					adjusted_output = w_filter.analysis(drop_frame_mode, win_config); //if win_config is true, win_step and length are flexible, else they are fixed to 8 12
					//-------------------------------------------

					auto t6 = chrono::high_resolution_clock::now();	//time statistics
					wfilter_time = chrono::duration_cast<chrono::microseconds>( t6 - t5).count();
				}

				auto t7 = chrono::high_resolution_clock::now();	//time statistics


				std::string expected_class = fn[frame_num];
				int first_idx = expected_class.find_last_of('_') + 1;
				expected_class = expected_class.substr(first_idx, expected_class.length()-4);
				expected_class.erase(expected_class.length()-4);

				std::cout << "-------------------------------------------------"<< endl;
				std::cout << "Frame num: " << frame_num << endl;
				std::cout << "Adjusted output: " << classes[adjusted_output] << endl;
				std::cout << "Expected class: " << expected_class << endl;
				std::cout << "-------------------------------------------------"<< endl;

				auto overall_time = chrono::duration_cast<chrono::microseconds>( t7 - t0 ).count();
				
				int n = w_filter.getwstep();
				time_arr.insert(time_arr.begin(), overall_time);
				if (time_arr.size() < n){
					n = time_arr.size();
				}
				float p_sum = 0.0f;
				for (int i = 0; i<n; i++){
					p_sum += time_arr[i];
				}
				//cout << "p_sum" << p_sum <<endl;
				if (time_arr.size() > 24){
					time_arr.pop_back();
				}

				std::string r_out, a_out;
				float cls_fps;
				if (c30){
					r_out = classes[output];
					a_out = classes[adjusted_output];
					display_output = classes[adjusted_output];
					display_roi = roi;

					// if (acc_time == 0){
					// 	acc_time = overall_time;
					// }

					acc_time += overall_time;
					cls_fps = 1000000/(float)acc_time;
					acc_time = 0;

					processed_frames += 1;

					//cout <<"expected" << expected_class <<" " << adjusted_output <<endl;
					if (expected_class == classes[output]){
						identified ++;
					}
					if (expected_class == classes[adjusted_output]){
						identified_adj++;
					}

					processing_fps = 1000000/p_sum;
					process_rate_arr.push_back(processing_fps);

				} else {
					r_out = " ";
					a_out = " ";
					acc_time += overall_time;
					cls_fps = 0;
					processing_fps = 0;
				}
				
				string u_stats = to_string(u[0]);
				string u_mode = to_string(u[4]);
				if (u[0] == 0){
					u_stats = "";
					u_mode = "";
				}

				myfile << frame_num << "," << cls_fps << "," << processing_fps << "," << r_out << "," << a_out << "," << cap_time << "," << preprocessing_time << "," <<  bnn_time << "," << wfilter_time << "," << uncertainty_time << "," <<  u_stats << "," << u[1] << "," << u[2] << "," << u[3] << "," << drop_frame_mode << "\n";
				
				if (frame_num != 0){
					total_time = total_time + (float)overall_time/1000000;
				}

				//Display output
				// rectangle(display_frame, roi, Scalar(0, 0, 255));
				putText(display_frame, classes[adjusted_output], Point(15, 55), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));	
				imshow("Original", display_frame);
				waitKey(25);

				frame_num++;

				char ESC = waitKey(1);	
				if (ESC == 27) 
				{
					cout << "ESC key is pressed by user" << endl;
					break;
				}	
			}

			float pro_sum = 0;
			for(float const &elem : process_rate_arr)
			{
				pro_sum += elem;
			}
			float processing_rate = pro_sum/process_rate_arr.size();
			//float processing_rate = 1/(wstep*(total_time/no_of_frame));

			float accuracy = 100.0*((float)identified/(float)processed_frames);
			float accuracy_adj = 100.0*((float)identified_adj/(float)processed_frames);
			float avg_cls_fps = (float)(processed_frames)/total_time;
			//float avg_rate = 1/((float)win_step*((float)total_time/(float)no_of_frame)); //avg rate for processing win_step number of frame
			cout << "results: " << identified << " " << processed_frames << " " << identified_adj << " " << total_time << " " << no_of_frame << endl;;
			myfile << "\n Accuracy, Adjusted Accuracy, Avg Classification Rate, Avg Processing Rate";
			myfile << "\n" << accuracy << "," << accuracy_adj << "," << avg_cls_fps << "," << processing_rate;
			myfile << "\n \n";
			myfile.close();

			float gain = 0;
			if (accuracy != 0){
				gain = (accuracy_adj-accuracy)*100/accuracy;
			}

			fs << folder_num << "," << wstep << "," << wlength << "," << accuracy << "," << accuracy_adj << "," << gain << "," << avg_cls_fps << "," << processing_rate << endl;;

			//----------------------------------------------------------------------------------------------------------------
			//----------------------------------------------------------------------------------------------------------------
			//----------------------------------------------------------------------------------------------------------------

		}
	}

	fs.close();
	//reset clock
	//config_clock(100);
    //Release memory
    sds_free(packedImages);
	sds_free(packedOut);
    return 1;
}