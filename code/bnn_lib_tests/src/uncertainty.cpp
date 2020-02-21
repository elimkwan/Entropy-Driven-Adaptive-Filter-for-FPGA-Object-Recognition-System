#include "uncertainty.hpp"

std::vector<double> Uncertainty::entropy_approach(std::vector<float> class_result, int mode){
    //mode 1: varience; mode 2:entropy

    std::vector<double> input_v(class_result.begin(), class_result.end());
    double ma = 0;

    double old_ma = __running_mean;
    double old_sd = __running_sd;

    double distribution; // varience or entropy of the input
    std::vector<double> pmf = softmax(input_v);

    if (mode == 1){
        distribution = init_var(pmf, 10)[2];
    }
    else if (mode == 2){
        distribution = entropy(pmf);
    }

    if (std::isnan(distribution)){
        cout << "entropy is nan" << endl;
        return {100, 100, 0, 0, 0};
    }

    insert_buf(__entropy_buf, distribution);
    int en_s = __entropy_buf.size();

    if (en_s < __lambda){
        cout << "----------initialising stage 1---------" << endl;
        set_dataSum(distribution);

        //print_vector(__entropy_buf);
        //print_vector(__ma_buf);
        return {distribution, 100, 0, 0, 1};

    }
    
    if (en_s == __lambda && __state == 0){
        cout << "----------initialising stage 2---------" << endl;
        ma = init_running_mean(distribution);
        insert_buf(__ma_buf, ma);

        //print_vector(__entropy_buf);
        //print_vector(__ma_buf);
        return {distribution, ma, 0, 0, 1};
    }

    if (__count < 50){
        ma = moving_avg(__entropy_buf,__lambda);
        __count += 1;
    } else{
        ma = naive_avg(__entropy_buf,__lambda);
        __count == 0;
    }

    insert_buf(__ma_buf, ma);
    int ma_s = __ma_buf.size();

    
    if (ma_s < __lambda){
        cout << "----------initialising stage 3---------" << endl;
        constraint_buf(__entropy_buf);

        //print_vector(__entropy_buf);
        //print_vector(__ma_buf);
        return {distribution, ma, 0, 0, 1};
    }

    if (ma_s == __lambda){
        cout << "----------initialising stage 4---------" << endl;
        vector <double> r;
        
        r = init_var(__ma_buf, __lambda);
        __mean_of_ma = r[0];
        __aggrM = r[1];

        constraint_buf(__entropy_buf);
        __state = 1;

        //print_vector(__entropy_buf);
        //print_vector(__ma_buf);
        return {distribution, ma, 0, 1, 1};
    }


    cout << "----------initialising stage 5(main loop)---------" << endl;

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

    int cur_state = update_state(ma, old_ma, old_sd, __alpha); //alpha = 2 for entropy
    int cur_mode = select_mode(3);
    constraint_buf(__entropy_buf);
    constraint_buf(__ma_buf);


    //print_vector(__entropy_buf);
    //print_vector(__ma_buf);
    return {distribution, ma, __running_sd, cur_state, cur_mode};

}

std::vector<double> Uncertainty::normalise(std::vector<double> &cp)
{	
    double mx = *max_element(std::begin(cp), std::end(cp));
    
    for(auto &elem : cp)
        elem = (double)elem / mx;
    
    return cp;
}

std::vector<double> Uncertainty::softmax(std::vector<double> &arg_vec)
{
    // Normalise the vector
    std::vector<double> norm_vec = normalise(arg_vec);
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

double Uncertainty::entropy(std::vector<double> &arg_vec)
{
    double sum = 0;
    for(auto const &elem : arg_vec){
        sum += elem * std::log2(1/elem);
    }
    //cout << "uncertainty: " << sum <<endl;
    return sum;
}

void Uncertainty::insert_buf(std::vector<double> &arg_vec, double &elem){
    arg_vec.insert (arg_vec.begin(), elem);

}

void Uncertainty::constraint_buf(std::vector<double> &arg_vec){
    if (arg_vec.size() > __lambda){
        arg_vec.pop_back();
    }
}



double Uncertainty::moving_avg(std::vector<double> &arg_vec, int n){
    double m =  __running_mean;
    __running_mean = m + (arg_vec[0]-arg_vec.back())/n;
    return __running_mean;
}

double Uncertainty::naive_avg(std::vector<double> &arg_vec, int n){
    double sum = 0;
    for(auto const &elem : arg_vec){
        sum += elem;
    }
    sum -= arg_vec.back();
    __running_mean = sum/n;
    return __running_mean;
}

vector<double> Uncertainty::init_var(vector<double> &ma, int n){
    cout<<"slow computation"<<endl;
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

vector<double> Uncertainty::moving_var(vector<double> ma, int n, double mean_of_ma, double arg_aggrM){

    //using welfard method
    cout<<"faster computation moving var"<<endl;
    cout<<"mean of ma" << mean_of_ma <<endl;
    //print_vector(ma);

    double old_mean = mean_of_ma;
    double new_mean = mean_of_ma + (ma[0]-ma.back())/n;

    arg_aggrM = arg_aggrM + (ma[0]-old_mean)*(ma[0]-new_mean) - (ma[n]-old_mean)*(ma[n]-new_mean);
    
    double newsd = sqrt(__aggrM/(n-1));

    return {new_mean, arg_aggrM, newsd};
}

int Uncertainty::update_state(double sample, double old_ma, double old_sd, float alpha){
    if (sample >= (old_ma-alpha*old_sd) && sample <= (old_ma+alpha*old_sd)){
        //cout << "upgrating mode: sample, old_ma, old_sd " << sample << " " << old_ma << " " << old_sd <<endl;
        if (__state != 18){
            __state += 1;
            return __state;
        }
    } else {
        __state = 1;
        return __state;
    }
}

int Uncertainty::select_mode(int n){
    if (__state <= n){
        return 1;
    }else if (__state <= (2*n + 1) ){
        return 2;
    }else if (__state <= (3*n + 3)){
        return 3;
    }else if (__state <= (4*n + 6)){
        return 4;
    }

}


void Uncertainty::set_dataSum(double &elem){
    __data_sum += elem;
}


double Uncertainty::init_running_mean(double &elem){
    __running_mean  = (__data_sum + elem)/ __lambda;
    return __running_mean;
}




//helper func: for printing vector
void Uncertainty::print_vector(std::vector<double> &vec)
{
	std::cout << "{ ";
	for(auto const &elem : vec)
	{
		std::cout << elem << " ";
	}
	std::cout << "}" <<endl;
}