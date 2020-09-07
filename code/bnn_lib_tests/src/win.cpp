/******************************************************************************
 * Code developed by Elim Kwan in April 2020 
 *
 * Window Filter
 * Generate aggregates of classification results.
 * Step Size and Length of Filter can be varied.
 * 
 *****************************************************************************/
#include "win.hpp"

unsigned int Win_filter::analysis(const std::vector<float> &class_result, int mode, bool flex, int aa, int bb, int cc, int dd, int ee, int ff, int gg, int hh, int ii, int jj){
/*
	Main Wrapper function of the Window Filter. 
	Stored recent classification result(the array class_result)and calculated the aggregate values based on that.
	Final adjusted output = Position of max element in aggregated class_result array. 

	@param mode: represents the level of uncertainty in data, depending on mode, various window filter configurations will be adopted if flex window setting is true
    @param flex: determines whether flex window filter setting is used.
    :return result_index: an integer representing a class(the adjusted output)
*/
	
	//cout << "Mode: " << mode << endl;
	if (!Win_filter::dropf()){
		Win_filter::update_memory(class_result);
	}

	//If base case is 15 fps, set to 9, if 30 fps, set to 4
	if (display_c == 4){
		display_f = true;
		display_c = 0;
	} else {
		display_f = false;
		display_c ++;
	}

	//choose window filter configurations
	if (flex && winit){
		unsigned int old_wstep = wstep;
		unsigned int old_wlength = wlength;
		// int new_config = 0;

		vector<int> r = select_ws_wl(mode, aa, bb, cc, dd, ee, ff, gg, hh, ii, jj);
		wstep = r[0];
		wlength = r[1];

		if (old_wlength != wlength || old_wstep != wstep){
			int a = (wstep <= wlength) ? wstep : wlength;
			wcount = a-1;

			//cout << "Changed Window Filter: " << wstep << " " << wlength << " ";
		}
	}

	//cout << "wcount: " << wcount << endl;
	//output real time result, when insufficient data to calculate aggregates
	int memory_size = wmemory.size();
	if (memory_size < wlength){
		//cout << "CASE 1: real time out" << endl;
		winit = false;
		int result_index = distance(wmemory[memory_size-1].begin(), max_element(wmemory[memory_size-1].begin(), wmemory[memory_size-1].end()));
		return result_index;
	}

	//Case A: When Step Size Less Than or Eual to Window Length 
	//		- cal at the end of step size - reset count at the end of step size
	//Case B: When Step Size Larger than Window Length
	//		- cal at the end of window length - reset count at the end of step size
	int k = (wstep <= wlength) ? wstep : wlength;

	//Check if finish initialisation
	if (!winit){
		winit = true;
		//cout << "k: " << k << endl;
		wcount = k-1;
	}

	//cout << "COUNT: " << wcount << endl;
	unsigned int win_out;
	if (wcount == (k-1)){
		//cout << "CASE 3: just calculated aggregated values" << endl;
		std::vector<float> adjusted_results(10, 0);

		for (int i=0; i<10; i++ ){
			for(int j = 0; j < wlength; j++){
				adjusted_results[i] += (wweights[j] * wmemory[j][i]);
			} 
		}
        win_out = distance(adjusted_results.begin(), max_element(adjusted_results.begin(), adjusted_results.end()));
        wpast_output = win_out; //update stored output
	} else {
		//cout << "CASE 2: stored output or CASE 4: dropping frame" << endl;
		win_out = wpast_output;
	}
	
	// if (mode == 5){
	// 	for(int i = 0; i<10; i++){
	// 		print_vector(wmemory[i]);
	// 	}
	// }

	wcount = (wcount == (wstep-1)) ? 0 : (wcount+1); // reset count at the end of step size 

	return win_out;
}



void Win_filter::update_memory(const std::vector<float> &class_result){
/*
	Update result history

	@para class_result: result of the cuurent frame, to be inserted to the memory array
*/
	std::vector<float> softmax_out = Win_filter::calculate_softmax(class_result);
	wmemory.push_back(softmax_out);
    if (wmemory.size() > max_wlength)
    {
        wmemory.erase(wmemory.begin());
    }
}

vector<float> Win_filter::calculate_softmax(const std::vector<float> &arg_vec){
/*
	Calculate softmax

	@para arg_vec: input vector with floating points
	:return: vector [e^(class1 probability)/sum, e^(class2 probability)/sum... e^(class10 probability)/sum], where sum = summation of e^(class probability) of all the classes
*/

	// Normalise the vector
    std::vector<float> norm_vec(arg_vec.begin(), arg_vec.end());
	int mx_n = *max_element(std::begin(norm_vec), std::end(norm_vec));
	if (mx_n == 0){return arg_vec; } //if class result is a zero vector, resturen zero softmax
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
		wweights[max_wlength-1-i] = expDecay(lambda, i);
        //std::cout << "init weights:" << wweights[i]<< endl;
	}
	// std::cout << "init weights:"<< endl;
	// print_vector(wweights);
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

bool Win_filter::dropf(){
/*
	Determine whether we should process the current frame. 
	If step size is larger than window length, and count is less than the difference between the two, we can save resources by not processing the frame  

	:return: boolean on whether to drop frame
*/
	//During init phase, not droping frames
	if (!winit){
		return false;
	}
	if (wstep > wlength && wcount >= wlength && wcount < wstep){
	//int k = (wstep <= wlength) ? wstep : wlength;
	//if (wcount == (k-1)){
		return true;
	}
	return false;
}

bool Win_filter::processf(){
/*
	Determine whether we should process the current frame. 
	If step size is larger than window length, and count is less than the difference between the two, we can save resources by not processing the frame  

	:return: boolean on whether to drop frame
*/
	//During init phase, not droping frames
	// if (!winit){
	// 	return true;
	// }
	//if (wstep > wlength && wcount >= wlength && wcount < wstep){
	int k = (wstep <= wlength) ? wstep : wlength;
	if (wcount == (k-1)){
		return true;
	}
	return false;
}

bool Win_filter::get_display_f(){
	
	return display_f;
}

vector<int> Win_filter::select_ws_wl(int mode, int aa, int bb, int cc, int dd, int ee, int ff, int gg, int hh, int ii, int jj){
/*
//int aa, int bb, int cc, int dd, int ee, int ff, int gg, int hh, int ii, int jj
	Return window filter configuration depending on current mode (for flex window feature)

	:return: window step size and length respectively
*/
	switch(mode) {

		// case 0: return {2,9};
		// case 1: return {2,9};
		// case 2: return {10,4};
		// case 3: return {5,9};
		// case 4: return {7,2};
		// case 5: return {3,2};

		// case 0: return {2,9};
		// case 1: return {2,9};
		// case 2: return {10,4};
		// case 3: return {5,9};
		// case 4: return {7,2};
		// case 5: return {5,2};

		case 1: return {aa,bb};
		case 2: return {cc,dd};
		case 3: return {ee,ff};
		case 4: return {gg,hh};
		case 5: return {ii,jj};

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