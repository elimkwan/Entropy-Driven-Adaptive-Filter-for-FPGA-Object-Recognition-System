/******************************************************************************
 * Code developed by Elim Kwan in April 2020 
 *
 * Uncertainty Estimation (Uncertainty Filter)
 * Calculate uncertainty in BNN output with: Entropy, Variance, AutoCorrelation 
 * 
 *****************************************************************************/
#include "uncertainty.hpp"

//------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------Main Wrapper Function-------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

std::vector<double> Uncertainty::cal_uncertainty(std::vector<float> class_result, string mode, int result){
/*
	Main wrapper function in uncertainty filter.
    "uncertainty_score" is the uncertainty score calculated from Entropy/Variance/AutoCorrelation

    @param class_result: an array containing the 10 class scores (current output from BNN)
    @param mode: "en/var/a" determines which uncertainty calculation schemes to be used
    @param result: raw output = Position of max element in the current class_result array
	:return {uncertainty_score, ma, __sd_of_uncertainty_score_running_mean, cur_state, cur_mode}: uncertainty_score, moving average of uncertainty_score, standard deviation of uncertainty_score, current state, current mode
*/

    if (mode == "na"){
        return {100, 100, 0, 0, 1};
    }

    //mode 1: varience; mode 2:entropy; mode 3: correlation
    srand(11); //set random seed for constant exp. result
    //when not using the uncertainty analysis at all

    std::vector<double> input_v(class_result.begin(), class_result.end());
    double uncertainty_score_runningmean = 0;

    double old_uncertainty_score_runningmean = __uncertainty_score_running_mean;
    double old_sd_of_uncertainty_score_running_mean = __sd_of_uncertainty_score_running_mean;

    double uncertainty_score; // correlation or varience or cal_entropy of the input
    std::vector<double> pmf = softmax(input_v);

    if (mode == "var"){
        uncertainty_score = cal_variance(pmf, 10)[2];
    } else if (mode == "en"){
        uncertainty_score = cal_entropy(pmf);
    } else if (mode == "a"){
        uncertainty_score = cal_autocorr(pmf, result);
    }

    if (std::isnan(uncertainty_score)){
        cout << "uncertainty score is nan" << endl;
        return {100, 100, 0, 0, 1};
    }

    insert_buf(__uncertainty_score_buf, uncertainty_score);
    int uncertainty_score_buf_size = __uncertainty_score_buf.size();

    if (uncertainty_score_buf_size < __lambda){
        //cout << "----------initialising stage 1---------" << endl;
        __uncertainty_score_sum += uncertainty_score;
        return {uncertainty_score, 100, 0, 0, 1};

    }
    
    if (uncertainty_score_buf_size == __lambda && __state == 0){
        //cout << "----------initialising stage 2---------" << endl;
        uncertainty_score_runningmean = running_mean_init(uncertainty_score);

        insert_buf(__uncertainty_score_runningmean_buf, uncertainty_score_runningmean);
        //cout << "Running Mean: " << uncertainty_score_runningmean << endl;
        //print_vector(__uncertainty_score_runningmean_buf);

        return {uncertainty_score, uncertainty_score_runningmean, 0, 0, 1};
    }

    //refresh uncertainty_score_runningmean: to sharpen the moving avaerage
    // if (__count < 50){
    //     uncertainty_score_runningmean = running_mean(__uncertainty_score_buf,__lambda);
    //     __count += 1;
    // } else{
    //     uncertainty_score_runningmean = naive_mean(__uncertainty_score_buf,__lambda);
    //     __count == 0;
    // }

    uncertainty_score_runningmean = running_mean(__uncertainty_score_buf,__lambda);
    insert_buf(__uncertainty_score_runningmean_buf, uncertainty_score_runningmean);
    int uncertainty_score_runningmean_buf_size = __uncertainty_score_runningmean_buf.size();

    //cout << "Running Mean" << uncertainty_score_runningmean << endl;
    //print_vector(__uncertainty_score_runningmean_buf);

    
    if (uncertainty_score_runningmean_buf_size < __lambda){
        //cout << "----------initialising stage 3---------" << endl;
        constraint_buf(__uncertainty_score_buf);
        return {uncertainty_score, uncertainty_score_runningmean, 0, 0, 1};
    }

    if (uncertainty_score_runningmean_buf_size == __lambda){
        //cout << "----------initialising stage 4---------" << endl;
        vector <double> r;
        
        r = cal_variance(__uncertainty_score_runningmean_buf, __lambda);
        __mean_of_uncertainty_score_runningmean = r[0];
        __aggrM = r[1];

        constraint_buf(__uncertainty_score_buf);
        __state = 1;
        return {uncertainty_score, uncertainty_score_runningmean, 0, 1, 1};
    }


    //cout << "----------initialising stage 5(main loop)---------" << endl;
    vector<double> a = running_var(__uncertainty_score_runningmean_buf, __lambda, __mean_of_uncertainty_score_runningmean, __aggrM);
    __mean_of_uncertainty_score_runningmean = a[0];
    __aggrM = a[1];
    __sd_of_uncertainty_score_running_mean = a[2];


    //init alpha
    if (__alpha == 0){
        //determine alpha
        int n = rand()%(__lambda);
        __alpha = abs(__uncertainty_score_runningmean_buf[__lambda]-__uncertainty_score_runningmean_buf[n])/__sd_of_uncertainty_score_running_mean;
        //cout<< "ALPHA SET: " << __alpha <<endl;
    }

    int aggressiveness = 3;

    int cur_state = update_state(uncertainty_score_runningmean, old_uncertainty_score_runningmean, old_sd_of_uncertainty_score_running_mean, __alpha, aggressiveness);
    int cur_mode = select_powersaving_mode(aggressiveness);
    constraint_buf(__uncertainty_score_buf);
    constraint_buf(__uncertainty_score_runningmean_buf);

    return {uncertainty_score, uncertainty_score_runningmean, __sd_of_uncertainty_score_running_mean, cur_state, cur_mode};

}


//------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------Method 1: Entropy-----------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

double Uncertainty::cal_entropy(std::vector<double> &arg_vec)
{
/*
	Entropy Calculation

    @param arg_vec: an array containing the 10 class scores (current output from BNN)
	:return sum: an integer representing the entropy of the class_scores array
*/
    double sum = 0;
    for(auto const &elem : arg_vec){
        sum += elem * std::log2(1/elem);
    }
    return sum;
}


//------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------Method 2: Variance----------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

vector<double> Uncertainty::cal_variance(vector<double> &ma, int n){
/*
	Variance Calculation

    @param ma: input array
    @param n: number of element in array
	:return {mean_of_ma, aggrM, (sum/(n-1))}: mean, aggregated square sum in variance equation, variance
*/
    double m = 0;
    for(auto const &elem : ma){
        //sum += pow((elem-mean),2);
        m += elem;
    }
    double mean_of_ma = m/n;

    double sum = 0;
    for(auto const &elem : ma){
        sum += pow((elem - mean_of_ma ),2);
    }
    double aggrM = sum;
    //cout << "INIT_VAR: updating mean_of_ma and aggrM " << mean_of_ma << " " << aggrM <<endl;
    //print_vector(ma);
    return {mean_of_ma, aggrM, (sum/(n-1))};
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------Method 3: Auto Correlation--------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
double Uncertainty::cal_autocorr(std::vector<double> &arg_vec, int result){
/*
	Wrapper for Autocorrelation Schemes

    __running_corr = {class1 autocorr result, class2 autocorr result ...}

    @param arg_vec: current class_scores array
    @param result: raw classification result (Index of max element in class_scores array)
	:return: an integer representing the uncertainty scores from Autocorrelation scheme
*/
    int n = 5;
    if (__corr_history.empty()){
        __corr_history = {{0.1}, {0.1}, {0.1}, {0.1}, {0.1}, {0.1}, {0.1}, {0.1}, {0.1}, {0.1}};
        //init_corr_history(); //Create the __corr_histroy 2D matrices
        __corr_init = false;
        return -1.0;
    }

    insert_corr_history(arg_vec); //Store class_score into __corr_history
    int num_of_stored_result = __corr_history[0].size(); //Check how many class_score array were stored
    //cout << "num_of_stored_result: " << num_of_stored_result << endl;

    if (num_of_stored_result < n){
        __corr_init = false;
        return -1.0;
    }

    if (num_of_stored_result == n && !__corr_init){
        for (int i = 0; i < 10; i++){
            __running_corr.push_back (running_autocorr_init(__corr_history[i]) );
        }
        __corr_init = true;

        //cout << "__running_cor: " << endl;
        //print_vector(__running_corr);
        return -1.0;
    }

    //double past_corr = __running_corr[result];
    for (int i = 0; i < 10; i++){
        __running_corr[i] = running_autocorr(__corr_history[i], __running_corr[i]);
    }

    // cout << "corr_history: " << endl;
    // for (int i = 0; i < __corr_history.size(); i++){
    //     print_vector(__corr_history[i]);
    // }
    // cout << "__running_cor: " << endl;
    // print_vector(__running_corr);

    constraint_corr_history(n);
    return __running_corr[result];
}

int Uncertainty::insert_corr_history(std::vector<double> &arg_vec){
/*
	Store class_scores arrays (prepare for autocorrelation calculation)

    @param arg_vec: input array
*/
    int s = arg_vec.size();

    for (int i = 0 ; i < s ; i++){
        double elem = arg_vec[i];
        if (elem != elem){return -1;}//check isnan
        __corr_history[i].insert(__corr_history[i].begin(), elem);
    }
    return 1;
}

void Uncertainty::constraint_corr_history(int n){
/*
	Discard oldest class_scores arrays from __corr_history

*/
    for (int i = 0 ; i < 10; i++){
        if (__corr_history[i].size() > n){
            __corr_history[i].pop_back();
        }
    }
}

double Uncertainty::running_autocorr(std::vector<double> &arg_vec, double old_corr){
/*
	Calculate running auto_correlation with partial sum (derived from definition)

    @param arg_vec: equals to __corr_history[result]
    @param result: raw classification result (Index of max element in class_scores array)
    :return new_correlation: new autocorrelation of the result class
*/
    int s = arg_vec.size();
    double new_correlation = old_corr + arg_vec[0]*arg_vec[1] - arg_vec[s-1]*arg_vec[s-2];
    return new_correlation;
}

double Uncertainty::running_autocorr_init(std::vector<double> &arg_vec){
/*
	Calculate Autocorrelation for the first time (initialisation phase, can't use running autocorrelation yet)

    @param arg_vec: equals to __corr_history[result]
    :return new_correlation: autocorrelation of the result class
*/
    double sum = 0;
    int s = arg_vec.size();
    for (int i = 0 ; i < (s-1) ; i++){
        sum += arg_vec[i]*arg_vec[i+1]; //assume padded 0
    }
    return sum;
}


//------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------Maths Functions-------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

double Uncertainty::naive_mean(std::vector<double> &arg_vec, int n){
/*
	Calculate Average of the input array

    @param arg_vec: input array
    @param n: size of input array
    :return __uncertainty_score_running_mean: an integer representing the mean of the input array
*/
    double sum = 0;
    for(auto const &elem : arg_vec){
        sum += elem;
    }
    sum -= arg_vec.back();
    __uncertainty_score_running_mean = sum/n;
    return __uncertainty_score_running_mean;
}

std::vector<double> Uncertainty::normalise(std::vector<double> &cp){
/*
	Normalised the input array

    @param cp: input array
    :return cp: normalised input array
*/
    double mx = *max_element(std::begin(cp), std::end(cp));
    
    for(auto &elem : cp)
        elem = (double)elem / mx;
    
    return cp;
}

std::vector<double> Uncertainty::softmax(std::vector<double> &arg_vec){
/*
	Apply softmax function to the input array

    @param arg_vec: input array
    :return norm_vec: output array containing predictive probabilities
*/
    // Normalise the vector
    std::vector<double> norm_vec = normalise(arg_vec);
    float mx = *max_element(std::begin(norm_vec), std::end(norm_vec));
    float sum = 0;
    for(auto const &elem : norm_vec)
        sum += exp(elem);
    
    if(sum == 0){
        std::cout << "Division by zero, sum = 0" << std::endl;
    }
    for(int i=0; i<10;i++)
    {
        norm_vec[i] = exp(norm_vec[i]) / sum;
    }

    return norm_vec;
}

void Uncertainty::insert_buf(std::vector<double> &arg_vec, double &elem){
/*
	Insert an element into an array

    @param arg_vec: input array
    @param elem: element to be inserted
*/
    arg_vec.insert (arg_vec.begin(), elem);

}

void Uncertainty::constraint_buf(std::vector<double> &arg_vec){
/*
	Insert an element into an array

    @param arg_vec: input array
    @param elem: element to be inserted
*/
    if (arg_vec.size() > __lambda){
        arg_vec.pop_back();
    }
}



double Uncertainty::running_mean(std::vector<double> &arg_vec, int n){
/*
	Calculating average with partial sum (referred to as moving average/rolling mean etc)

    @param arg_vec: input array
    @param n: size of input array
    :return __uncertainty_score_running_mean: new moving averages
*/
    double m =  __uncertainty_score_running_mean;
    __uncertainty_score_running_mean = m + (arg_vec[0]-arg_vec.back())/n;
    return __uncertainty_score_running_mean;
}

vector<double> Uncertainty::running_var(vector<double> ma, int n, double mean_of_ma, double arg_aggrM){
/*
	Calculating variance with partial sum, using welford method

    @param ma: input array
    @param n: size of input array
    @param mean_of_ma: mean of input array
    @param arg_aggrM: aggregated square root sum in variance equation
    :return {new_mean, arg_aggrM, newsd}: new mean, new partial sum, new standard deviation
*/
    double old_mean = mean_of_ma;
    double new_mean = mean_of_ma + (ma[0]-ma.back())/n;

    arg_aggrM = arg_aggrM + (ma[0]-old_mean)*(ma[0]-new_mean) - (ma[n]-old_mean)*(ma[n]-new_mean);
    
    double newsd = sqrt(__aggrM/(n-1));

    return {new_mean, arg_aggrM, newsd};
}

int Uncertainty::update_state(double score, double old_ma, double old_sd, float alpha, int n){
/*
	Update internal state based on whether the current uncertainty score is within the range 
    range: (old_ma - alpha * old_sd) <= uncertainty score <=  (old_ma + alpha * old_sd)

    @param score: moving average of uncertainty score
    @param old_ma: moving average of the previous frame
    @param old_sd: standard deviation of the previous frame
    @param alpha: arbitary parameters for determining the range
    @param n: maximium state
    :return __state: state of the object
*/
    // if (__state <= 15){
    //     __state += 1;
    //     return __state;
    // } else 
    if (score >= (old_ma-alpha*old_sd) && score <= (old_ma+alpha*old_sd)) {
        if (__state != 15){
            __state += 1;
        }
        return __state;
    } else {
        __state = 1;
        return __state;
    }
}

int Uncertainty::select_powersaving_mode(int n){
/*
	Classify the current frame into 5 levels (that will be output to the main program), based on their current states
    Mode 1: least certain -> Mode 5: very certain
    State: ranges from 0 to 15, act as progression counter within the class member
    y = 3 log10(x) + 1

    @param n: an arbitary integer, increase it if we prefer more internal states
    :return: an integer indiating the mode
*/
    if (__state <= 1){
        return 1;
    }else if (__state <= 3){
        return 2;
    }else if (__state <= 6){
        return 3;
    }else if (__state <= 14){
        return 4;
    }else{
        return 5;
    }

}


// void Uncertainty::set_dataSum(double &elem){
// /*
// 	Increment partial sum (update __uncertainty_score_sum)

//     @param elem: number to be added
// */
//     __uncertainty_score_sum += elem;
// }


double Uncertainty::running_mean_init(double &elem){
/*
	Calculate average for the first time with __uncertainty_score_sum (cannot use rolling mean yet during initialisation phase)

    @param elem: number to be added
    :return __uncertainty_score_running_mean: mean (bases of rolling mean calculation)
*/
    __uncertainty_score_running_mean  = (__uncertainty_score_sum + elem)/ __lambda;
    return __uncertainty_score_running_mean;
}


void Uncertainty::print_vector(std::vector<double> &vec){
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

