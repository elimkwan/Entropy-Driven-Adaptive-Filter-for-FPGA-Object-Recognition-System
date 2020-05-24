/******************************************************************************
 * Code developed by Elim Kwan in April 2020 
 *
 * Window Filter
 * Generate aggregates of classification results.
 * Step Size and Length of Filter can be varied.
 * 
 *****************************************************************************/
#include "win.hpp"

unsigned int Win_filter::analysis(int mode, bool flex){
/*
	Main Wrapper function of the Window Filter. 
	Stored recent classification result(the array class_result)and calculated the aggregate values based on that.
	Final adjusted output = Position of max element in aggregated class_result array. 

	@param mode: represents the level of uncertainty in data, depending on mode, various window filter configurations will be adopted if flex window setting is true
    @param flex: determines whether flex window filter setting is used.
    :return result_index: an integer representing a class(the adjusted output)
*/
	
	if (flex){
		unsigned int old_wstep = wstep;
		unsigned int old_wlength = wlength;
		int new_config = 0;

		vector<int> r = select_ws_wl(mode);
		wstep = r[0];
		wlength = r[1]; 

		if (old_wlength != wlength || old_wstep > wstep){
			wcount = wstep; //set count to new step size
			new_config = 1;
		}
	}
	
	int num_histroy = wmemory.size();

	int result_index = distance(wmemory[0].begin(), max_element(wmemory[0].begin(), wmemory[0].end()));

	if (num_histroy <= wstep || num_histroy < wlength){
		//not enough data for previous analysis, return real time data
		return result_index;
	}

	if (wcount < wstep){
        wcount++;
		return wpast_output;
	} else if ( wcount == wstep){
		std::vector<float> adjusted_results(10, 0);
		for(int i = 0; i < wlength; i++)
		{ 
			for(int j = 0; j < 10; j++)
			{
				adjusted_results[j] += (wweights[i] * wmemory[i][j]);
			}
		}
        unsigned int adjusted_output = distance(adjusted_results.begin(), max_element(adjusted_results.begin(), adjusted_results.end()));
        
        wcount = 0;//reset count
        wpast_output = adjusted_output;

		return adjusted_output;
	}

	//Error
	return result_index;
}



void Win_filter::update_memory(const std::vector<float> &class_result){
/*
	Update result_history

	@para class_result: result of the cuurent frame, to be inserted to the history array
*/
	
    wmemory.insert(wmemory.begin(), Win_filter::calculate_certainty(class_result));
    if (wmemory.size() > max_wlength)
    {
        wmemory.pop_back();
    }
}

vector<float> Win_filter::calculate_certainty(const std::vector<float> &arg_vec){
/*
	Calculate the aggregates

	@para arg_vec: input vector with floating points
	:return: vector [e^(class1 probability)/sum, e^(class2 probability)/sum... e^(class10 probability)/sum], where sum = summation of e^(class probability) of all the classes
*/

	// Normalise the vector
    std::vector<float> norm_vec(arg_vec.begin(), arg_vec.end());
	int mx_n = *max_element(std::begin(norm_vec), std::end(norm_vec));
	for(auto &elem : norm_vec)
		elem = (float)elem / mx_n;

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

void Win_filter::init_weights(float lambda){
/*
	Initialise weights of Window Filter

	@param lambda: set to 0.2 (parameters probed by Musab)
*/
    for(int i = 0; i < max_wlength; i++)
	{
		wweights[i] = expDecay(lambda, i);
        std::cout << "init weights:" << wweights[i]<< endl;
	}
}

float Win_filter::expDecay(float lambda, int t, int N){
/*
	Exponential Decay Function

	@param lambda: set to 0.2 (parameters probed by Musab)
	@param t: event time unit
	@param N: default to 1 (another arbitary parameter that is not used)
	:return: the y value of the exponential decay curve given x (event time)
*/
	return N * std::exp(-(lambda * t));
}

int Win_filter::getwstep(){
/*
	Return window step size

	:return wstep: step size of window filter
*/
	return wstep;
}

vector<int> Win_filter::select_ws_wl(int mode){
/*
	Return window filter configuration depending on current mode (for flex window feature)

	:return: window step size and length respectively
*/
	if (mode == 0 || mode == 1){
		return {1,24};
	} else if (mode == 5 || mode == 2 || mode == 3 || mode == 4 ){
		return {4,24};
	}
}

void Win_filter::print_vector(std::vector<float> &vec)
{
/*
	Printing vector

	@param &vec: Vector to be printed
*/
	std::cout << "{ ";
	for(auto const &elem : vec)
	{
		std::cout << elem << " ";
	}
	std::cout << "}" <<endl;
}