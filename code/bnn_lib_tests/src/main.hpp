// #include "../tiny_cnn/tiny_cnn.h"
// #include "../tiny_cnn/util/util.h"
// #include "foldedmv-offload.h"
#include <iostream>
//#include <math.h> 

using namespace std;
using namespace cv;


namespace basic
{
	double clockToMilliseconds(clock_t ticks){
    	// units/(units/time) => time (seconds) * 1000 = milliseconds
    	return (ticks/(double)CLOCKS_PER_SEC)*1000.0;
	}

	// std::vector<float> normalise(std::vector<float> &cp)
	// {	
	// 	float mx = *max_element(std::begin(cp), std::end(cp));
		
	// 	for(auto &elem : cp)
	// 		elem = (float)elem / mx;
		
	// 	return cp;
	// }

	// vector<float> softmax(std::vector<float> &arg_vec)
	// {
	// 	// Normalise the vector
	// 	std::vector<float> norm_vec = normalise(arg_vec);
	// 	float mx = *max_element(std::begin(norm_vec), std::end(norm_vec));
	// 	float sum = 0;
	// 	for(auto const &elem : norm_vec)
	// 		sum += exp(elem);
		
	// 	if(sum == 0){
	// 		std::cout << "Division by zero, sum = 0" << std::endl;
	// 	}
	// 	// Try to use OpenMP
	// 	for(int i=0; i<10;i++)
	// 	{
	// 		norm_vec[i] = exp(norm_vec[i]) / sum;
	// 	}

	// 	// cout<< "Probability list" <<endl;
	// 	// print_vector(norm_vec);

	// 	return norm_vec;
	// }

	// float entropy(std::vector<float> &arg_vec)
	// {
	// 	float sum = 0;
	// 	for(auto const &elem : arg_vec){
	// 		sum += elem * std::log2(1/elem);
	// 	}
	// 	//cout << "uncertainty: " << sum <<endl;
	// 	return sum;
	// }

	// float cross_entropy(std::vector<float> &p1, std::vector<float> &p2)
	// {
	// 	float sum = 0;
	// 	int i = 0;
	// 	for(auto const &elem : p1){
	// 		sum += elem * std::log2(elem/p2[i]);
	// 		i++;
	// 	}
	// 	//cout << "uncertainty: " << sum <<endl;
	// 	return sum;
	// }

	// float sd(std::vector<float> &arg_vec)
	// {	
	// 	float mean = 0;
	// 	int size = arg_vec.size();
	// 	for(auto const &elem : arg_vec){
	// 		mean += elem;
	// 	}
	// 	mean = mean/size;

	// 	float sum = 0;
	// 	for(auto const &elem : arg_vec){
	// 		sum += pow((elem-mean),2);
	// 	}

	// 	return sqrt(sum/size);
	// }


}


// template<typename T>
// class basic_func
// {
// 	public:

// 		basic_func(){}
// 		inline void print_vector(std::vector<T> &vec);
// 		double clockToMilliseconds(clock_t ticks);
// 		float expDecay(T lambda, int t, int N = 1);
// 		void flatten_mat(cv::Mat &m, std::vector<T> &v);
// 		inline std::vector<float> normalise(std::vector<T> &vec);
// 		vector<float> softmax(std::vector<T> &arg_vec);
// 		float entropy(std::vector<T> &arg_vec);

// };

// template<typename T>
// inline void basic_func::print_vector(std::vector<T> &vec)
// {
// 	cout << "-------------------------------------" << endl;
// 	std::cout << "{ ";
// 	for(auto const &elem : vec)
// 	{
// 		std::cout << elem << " ";
// 	}
// 	std::cout << "}" <<endl;
// 	cout << "-------------------------------------" << endl;
// }



// template<typename T>
// float basic_func::expDecay(T lambda, int t, int N = 1)
// {
// 	// Remove N if it is not needed
// 	return N * std::exp(-(lambda * (T)t));
// }


// Convert matrix into a vector 
// |1 0 0|
// |0 1 0| -> [1 0 0 0 1 0 0 0 1]
// |0 0 1|
// template<typename T>
// void basic_func::flatten_mat(cv::Mat &m, std::vector<T> &v)
// {
// 	if(m.isContinuous()) 
// 	{
// 		//cout<< "data is continuous"<< endl;
// 		v.assign(m.datastart, m.dataend);
// 	} 
// 	else 
// 	{
// 		cout<< "data is not continuous"<< endl;
// 		for (int i = 0; i < m.rows; ++i) 
// 		{
// 			v.insert(v.end(), m.ptr<T>(i), m.ptr<T>(i)+m.cols);
// 		}
// 	}
// }

// template<typename T>
// inline std::vector<float> basic_func::normalise(std::vector<T> &vec)
// {	
// 	std::vector<float> cp(vec.begin(), vec.end());
// 	T mx = *max_element(std::begin(cp), std::end(cp));
	
// 	for(auto &elem : cp)
// 		elem = (float)elem / mx;
	
// 	return cp;
// }

/*
	Calculate certainty

	@para arg_vec: input vector with floating points
	@return vector [e^(class1 probability)/sum, e^(class2 probability)/sum... e^(class10 probability)/sum], where sum = summation of e^(class probability) of all the classes
*/
// template<typename T>
// vector<float> basic_func::softmax(std::vector<T> &arg_vec)
// {
// 	// Normalise the vector
// 	std::vector<float> norm_vec = normalise(arg_vec);
// 	float mx = *max_element(std::begin(norm_vec), std::end(norm_vec));
// 	float sum = 0;
// 	for(auto const &elem : norm_vec)
// 		sum += exp(elem);
	
// 	if(sum == 0){
// 		std::cout << "Division by zero, sum = 0" << std::endl;
// 	}
// 	// Try to use OpenMP
// 	for(int i=0; i<10;i++)
// 	{
// 		norm_vec[i] = exp(norm_vec[i]) / sum;
// 	}

// 	cout<< "Probability list" <<endl;
// 	print_vector(norm_vec);

// 	return norm_vec;
// }

// template<typename T>
// float basic_func::entropy(std::vector<T> &arg_vec)
// {
// 	float sum = 0;
// 	for(auto const &elem : arg_vec){
// 		sum += elem * std::log(1/elem);
// 	}
// 	cout << "uncertainty: " << sum <<endl;
// 	return sum;
// }

/*
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
*/