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
    "distribution" is the uncertainty score calculated from Entropy/Variance/AutoCorrelation

    @param class_result: an array containing the 10 class scores (current output from BNN)
    @param mode: "en/var/a" determines which uncertainty calculation schemes to be used
    @param result: raw output = Position of max element in the current class_result array
	:return {distribution, ma, __running_sd, cur_state, cur_mode}: distribution, moving average of distribution, standard deviation of distribution, current state, current mode
*/

    //mode 1: varience; mode 2:entropy; mode 3: correlation
    srand(11); //set random seed for constant exp. result
    //when not using the uncertainty analysis at all
    if (mode == "na"){
        return {100, 100, 0, 0, 0};
    }

    std::vector<double> input_v(class_result.begin(), class_result.end());
    double ma = 0;

    double old_ma = __running_mean;
    double old_sd = __running_sd;

    double distribution; // varience or entropy of the input
    std::vector<double> pmf = softmax(input_v);

    if (mode == "var"){
        distribution = init_var(pmf, 10)[2];
    }
    else if (mode == "en"){
        distribution = entropy(pmf);
    } else if (mode == "a"){
        distribution = auto_corr_wrapper(pmf, result);
    }

    if (std::isnan(distribution)){
        cout << "entropy is nan" << endl;
        return {100, 100, 0, 0, 0};
    }

    insert_buf(__entropy_buf, distribution);
    int en_s = __entropy_buf.size();

    if (en_s < __lambda){
        //cout << "----------initialising stage 1---------" << endl;
        set_dataSum(distribution);
        return {distribution, 100, 0, 0, 1};

    }
    
    if (en_s == __lambda && __state == 0){
        //cout << "----------initialising stage 2---------" << endl;
        ma = init_running_mean(distribution);
        insert_buf(__ma_buf, ma);
        return {distribution, ma, 0, 0, 1};
    }

    ma = moving_avg(__entropy_buf,__lambda);

    //refresh ma: to sharpen the moving avaerage
    // if (__count < 50){
    //     ma = moving_avg(__entropy_buf,__lambda);
    //     __count += 1;
    // } else{
    //     ma = naive_avg(__entropy_buf,__lambda);
    //     __count == 0;
    // }

    insert_buf(__ma_buf, ma);
    int ma_s = __ma_buf.size();

    
    if (ma_s < __lambda){
        //cout << "----------initialising stage 3---------" << endl;
        constraint_buf(__entropy_buf);

        //print_vector(__entropy_buf);
        //print_vector(__ma_buf);
        return {distribution, ma, 0, 0, 1};
    }

    if (ma_s == __lambda){
        //cout << "----------initialising stage 4---------" << endl;
        vector <double> r;
        
        r = init_var(__ma_buf, __lambda);
        __mean_of_ma = r[0];
        __aggrM = r[1];

        constraint_buf(__entropy_buf);
        __state = 1;
        return {distribution, ma, 0, 1, 1};
    }


    //cout << "----------initialising stage 5(main loop)---------" << endl;
    vector<double> a = moving_var(__ma_buf, __lambda, __mean_of_ma, __aggrM);
    __mean_of_ma = a[0];
    __aggrM = a[1];
    __running_sd = a[2];


    //init alpha
    if (__alpha == 0){
        //determine alpha
        int n = rand()%(__lambda);
        __alpha = abs(__ma_buf[__lambda]-__ma_buf[n])/__running_sd;
        cout<< "ALPHA SET: " << __alpha <<endl;
    }

    int aggressiveness = 3;

    int cur_state = update_state(ma, old_ma, old_sd, __alpha, aggressiveness);
    int cur_mode = select_mode(aggressiveness);
    constraint_buf(__entropy_buf);
    constraint_buf(__ma_buf);

    return {distribution, ma, __running_sd, cur_state, cur_mode};

}


//------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------Method 1: Entropy-----------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

double Uncertainty::entropy(std::vector<double> &arg_vec)
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

vector<double> Uncertainty::init_var(vector<double> &ma, int n){
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
    cout << "INIT_VAR: updating mean_of_ma and aggrM " << mean_of_ma << " " << aggrM <<endl;
    //print_vector(ma);
    return {mean_of_ma, aggrM, (sum/(n-1))};
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------Method 3: Auto Correlation--------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
double Uncertainty::auto_corr_wrapper(std::vector<double> &arg_vec, int result){
/*
	Wrapper for Autocorrelation Schemes

    @param arg_vec: current class_scores array
    @param result: raw classification result (Index of max element in class_scores array)
	:return: an integer representing the uncertainty scores from Autocorrelation scheme
*/
    if (__corr_history.empty()){
        init_corr_history(arg_vec);
        return -1.0;
    }

    insert_corr_history(arg_vec);
    int s = __corr_history[0].size();

    if (s < 4){
        return -1.0;
    }

    if (s == 4){
        for (int i = 9; i > -1; i--){
            __running_corr.push_back (init_auto_correlation(__corr_history[i]) );
        }
        return -1.0;
    }
    double past_corr = __running_corr[result];
    for (int i = 0; i < 10; i++){
        __running_corr[i] = running_auto_correlation(__corr_history[i], i);
    }
    return __running_corr[result];
}

void Uncertainty::init_corr_history(std::vector<double> &arg_vec){
/*
	Initialise the size of __corr_history array(prepare for autocorrelation calculation)

    @param arg_vec: input array
*/
    std::vector <double> emptyrol (1, 0.0);
    for(int i=0; i<10; i++)
    {
        __corr_history.push_back(emptyrol);
    }
}

void Uncertainty::insert_corr_history(std::vector<double> &arg_vec){
/*
	Store class_scores arrays (prepare for autocorrelation calculation)

    @param arg_vec: input array
*/
    int s = arg_vec.size();
    for (int i = 0 ; i < s ; i++){
        __corr_history[i].insert(__corr_history[i].begin(), arg_vec[i]);
    }
}

void Uncertainty::constraint_corr_history(){
/*
	Discard oldest class_scores arrays from __corr_history

*/
    for (int i = 0 ; i < 10; i++){
        if (__corr_history[i].size() > 4){
            __corr_history[i].pop_back();
        }
    }
}

double Uncertainty::running_auto_correlation(std::vector<double> &arg_vec, int result){
/*
	Calculate running auto_correlation with partial sum (derived from definition)

    @param arg_vec: equals to __corr_history[result]
    @param result: raw classification result (Index of max element in class_scores array)
    :return new_correlation: new autocorrelation of the result class
*/
    double new_correlation = __running_corr[result] + arg_vec[0]*arg_vec[1] - arg_vec[4]*arg_vec[3];
    return new_correlation;
}

double Uncertainty::init_auto_correlation(std::vector<double> &arg_vec){
/*
	Calculate Autocorrelation for the first time (initialisation phase, can't use running autocorrelation yet)

    @param arg_vec: equals to __corr_history[result]
    :return new_correlation: autocorrelation of the result class
*/
    double sum = 0;
    int s = arg_vec.size();
    for (int i = 0 ; i < s-1 ; i++){
        sum += arg_vec[i]*arg_vec[i+1];
    }
    return (sum/s);
}


//------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------Maths Functions-------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

double Uncertainty::naive_avg(std::vector<double> &arg_vec, int n){
/*
	Calculate Average of the input array

    @param arg_vec: input array
    @param n: size of input array
    :return __running_mean: an integer representing the mean of the input array
*/
    double sum = 0;
    for(auto const &elem : arg_vec){
        sum += elem;
    }
    sum -= arg_vec.back();
    __running_mean = sum/n;
    return __running_mean;
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



double Uncertainty::moving_avg(std::vector<double> &arg_vec, int n){
/*
	Calculating average with partial sum (referred to as moving average/rolling mean etc)

    @param arg_vec: input array
    @param n: size of input array
    :return __running_mean: new moving averages
*/
    double m =  __running_mean;
    __running_mean = m + (arg_vec[0]-arg_vec.back())/n;
    return __running_mean;
}

vector<double> Uncertainty::moving_var(vector<double> ma, int n, double mean_of_ma, double arg_aggrM){
/*
	Calculating variance with partial sum, using welfard method

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

int Uncertainty::update_state(double sample, double old_ma, double old_sd, float alpha, int n){
/*
	Calculating variance with partial sum, using welfard method

    @param ma: input array
    @param n: size of input array
    @param mean_of_ma: mean of input array
    @param arg_aggrM: aggregated square root sum in variance equation
    :return {new_mean, arg_aggrM, newsd}: new mean, new partial sum, new standard deviation
*/
    if (__state <= n){
        __state += 1;
        return __state;
    } else if (sample >= (old_ma-alpha*old_sd) && sample <= (old_ma+alpha*old_sd)){
        if (__state != (4*n +4)){
            __state += 1;
        }
        return __state;
    } else {
        __state = 1;
        return __state;
    }
}

int Uncertainty::select_mode(int n){
/*
	Classify the current frame into 5 levels (that will be output to the main program), based on their current states
    Mode 1: least certain -> Mode 5: very certain
    State: ranges from 0 to 18, act as progression counter within the class member

    @param n: an arbitary integer, increase it if we prefer more internal states
    :return: an integer indiating the mode
*/
    if (__state <= n){
        return 1;
    }else if (__state <= (2*n) ){
        return 2;
    }else if (__state <= (3*n + 1)){
        return 3;
    }else if (__state < (4*n + 3)){
        return 4;
    }else{
        return 5;
    }

}


void Uncertainty::set_dataSum(double &elem){
/*
	Increment partial sum (update __data_sum)

    @param elem: number to be added
*/
    __data_sum += elem;
}


double Uncertainty::init_running_mean(double &elem){
/*
	Calculate average for the first time with __data_sum (cannot use rolling mean yet during initialisation phase)

    @param elem: number to be added
    :return __running_mean: mean (bases of rolling mean calculation)
*/
    __running_mean  = (__data_sum + elem)/ __lambda;
    return __running_mean;
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

