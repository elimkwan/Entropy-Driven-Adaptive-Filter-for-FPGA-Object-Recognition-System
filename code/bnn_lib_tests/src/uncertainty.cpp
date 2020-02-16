#include "uncertainty.hpp"

std::vector<double> Uncertainty::wrapper(std::vector<float> class_result){

    std::vector<double> input_v(class_result.begin(), class_result.end());
    double ma = 0;

    double old_ma = __running_mean;
    double old_sd = __running_sd;

    std::vector<double> pmf = softmax(input_v);
    double cur_entropy = entropy(pmf);

    if (std::isnan(cur_entropy)){
        cout << "entropy is nan" << endl;
        return {100, 100, 0, 0, 0};
    }

    insert_buf(__entropy_buf, cur_entropy);
    int en_s = __entropy_buf.size();

    if (en_s < __lambda){
        cout << "----------initialising stage 1---------" << endl;
        set_dataSum(cur_entropy);

        //print_vector(__entropy_buf);
        //print_vector(__ma_buf);
        return {cur_entropy, 100, 0, 0, 1};

    }
    
    if (en_s == __lambda && __state == 0){
        cout << "----------initialising stage 2---------" << endl;
        ma = init_running_mean(cur_entropy);
        insert_buf(__ma_buf, ma);

        //print_vector(__entropy_buf);
        //print_vector(__ma_buf);
        return {cur_entropy, ma, 0, 0, 1};
    }

    if (__count < 30){
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
        return {cur_entropy, ma, 0, 0, 1};
    }

    if (ma_s == __lambda){
        cout << "----------initialising stage 4---------" << endl;
        init_var(__ma_buf, __lambda);
        constraint_buf(__entropy_buf);
        __state = 1;


        //print_vector(__entropy_buf);
        //print_vector(__ma_buf);
        return {cur_entropy, ma, 0, 0, 1};
    }

    cout << "----------initialising stage 5(main loop)---------" << endl;

    __running_sd = moving_var(__ma_buf, __lambda);
    int cur_state = update_state(ma, old_ma, old_sd, 0.5);
    int cur_mode = select_mode(3);
    constraint_buf(__entropy_buf);
    constraint_buf(__ma_buf);


    //print_vector(__entropy_buf);
    //print_vector(__ma_buf);
    return {cur_entropy, ma, __running_sd, cur_state, cur_mode};

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
    double __running_mean = __running_mean + (arg_vec[0]-arg_vec.back())/n;
    return __running_mean;
}

double Uncertainty::naive_avg(std::vector<double> &arg_vec, int n){
    double sum = 0;
    for(auto const &elem : arg_vec){
        sum += elem;
    }
    __running_mean = sum/n;
    return __running_mean;
}

double Uncertainty::init_var(vector<double> ma, int n){
    cout<<"slow computation"<<endl;
    double m = 0;
    for(auto const &elem : ma){
        //sum += pow((elem-mean),2);
        m += elem;
    }
    __mean_of_ma = m/n;

    double sum = 0;
    for(auto const &elem : ma){
        sum += pow((elem - __mean_of_ma ),2);
    }
    __aggrM = sum;
    cout << "updating mean_of_ma and aggrM " << __mean_of_ma << " " << __aggrM <<endl;
    print_vector(ma);
    return sqrt(sum/(n-1));
}

double Uncertainty::moving_var(vector<double> ma, int n){

    //using welfard method
    cout<<"faster computation"<<endl;

    double old_mean = __mean_of_ma;
    double new_mean = __mean_of_ma + (ma[0]-ma.back())/n;
    __mean_of_ma = new_mean;

    __aggrM = __aggrM + (ma[0]-old_mean)*(ma[0]-new_mean) - (ma[n]-old_mean)*(ma[n]-new_mean);
    
    double newsd = sqrt(__aggrM/(n-1));

    return newsd;
}

int Uncertainty::update_state(double sample, double old_ma, double old_sd, double alpha){
    if (sample >= (old_ma-alpha*old_sd) && sample <= (old_ma+alpha*old_sd)){
        cout << "upgrating mode: sample, old_ma, old_sd " << sample << " " << old_ma << " " << old_sd <<endl;
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