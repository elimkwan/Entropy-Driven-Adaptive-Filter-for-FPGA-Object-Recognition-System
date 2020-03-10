#include "win.hpp"

//helper func: for printing vector
void Win_filter::print_vector(std::vector<float> &vec)
{
	std::cout << "{ ";
	for(auto const &elem : vec)
	{
		std::cout << elem << " ";
	}
	std::cout << "}" <<endl;
}

vector<int> Win_filter::select_ws_wl(int mode){
	if (mode == 0 || mode == 1 || mode == 2){
		return {4,12};
	} else if (mode == 3 || mode == 4){
		return {8,12};
	} else if (mode == 5) {
		return {10, 12};
	}
}

//return the adjusted output
unsigned int Win_filter::analysis(int mode, bool flex){
    //win_step, step_counts, past_output, results_history, weights
    //cout << "output filer starts here ..." << endl;
	
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

		//cout << "new config, wcount, window step size and length: " << new_config << " " << wcount << " " << wstep << " " << wlength <<endl;
	}
	
	int num_histroy = wmemory.size();

	// for (int i = 0; i < wmemory.size(); i++)
	// {
	// 	print_vector(wmemory[i]);
	// }

	int result_index = distance(wmemory[0].begin(), max_element(wmemory[0].begin(), wmemory[0].end()));

	if (num_histroy <= wstep || num_histroy < wlength){
		//not enough data for previous analysis, return real time data
		//cout << "outputing real time data" << endl;
		//std::cout << "arg_count: " << arg_count << std::endl;
		//std::cout << "arg_results_history" << std::endl;
		// for (int i = 0; i < arg_results_history.size(); i++)
		// {
		// 	print_vector(arg_results_history[i]);
		// }

		//std::vector<float> current_result = arg_results_history[arg_count];
		//float result_value = 0.00f;
		//result_value = std::max_element(current_result.begin(), current_result.end());
		//std::cout << "real time data results index: " << result_index << std::endl;
		//print_vector(arg_results_history[0]);
		return result_index;
	}

	if (wcount < wstep){
		//cout << "outputing wcount < wstep data" << endl;
        wcount++;
		return wpast_output;
	} else if ( wcount == wstep){

		//cout << "outputing wcount == wstep, current analysised result" << endl;
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

	//cout << "should not end up here - output non adjusted data as replacement" << endl;
	return result_index;
}




//update result_history
void Win_filter::update_memory(const std::vector<float> &class_result){
	cout << "update memory" << endl;
    wmemory.insert(wmemory.begin(), Win_filter::calculate_certainty(class_result));
    if (wmemory.size() > max_wlength)
    {
        wmemory.pop_back();
    }
}

/*
	Calculate certainty

	@para arg_vec: input vector with floating points
	@return vector [e^(class1 probability)/sum, e^(class2 probability)/sum... e^(class10 probability)/sum], where sum = summation of e^(class probability) of all the classes
*/
vector<float> Win_filter::calculate_certainty(const std::vector<float> &arg_vec)
{
	// Normalise the vector
    std::vector<float> norm_vec(arg_vec.begin(), arg_vec.end());
	int mx_n = *max_element(std::begin(norm_vec), std::end(norm_vec));
	for(auto &elem : norm_vec)
		elem = (float)elem / mx_n;

	//std::vector<float> norm_vec = normalise(arg_vec);
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

// std::vector<float> Win_filter::normalise(std::vector<float> &vec)
// {	
// 	std::vector<float> cp(vec.begin(), vec.end());
// 	int mx = *max_element(std::begin(cp), std::end(cp));
	
// 	for(auto &elem : cp)
// 		elem = (float)elem / mx;
	
// 	return cp;
// }



//initialising weights
void Win_filter::init_weights(float lambda){
    for(int i = 0; i < max_wlength; i++)
	{
		wweights[i] = expDecay(lambda, i);
        std::cout << "init weights:" << wweights[i]<< endl;
	}
}

float Win_filter::expDecay(float lambda, int t, int N)
{
	return N * std::exp(-(lambda * t));
}