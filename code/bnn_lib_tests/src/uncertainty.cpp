#include "uncertainty.hpp"

std::vector<float> Uncertainty::wrapper(std::vector<float> class_result){

    std::vector<float> pmf = softmax(class_result);
    float cur_entropy = entropy(pmf);

    if (isnan(cur_entropy)){
        cout << "entropy is nan" << endl;
        //return {100.f, 0.f, 0.f,0.f, 0.f, 0.f};
        return {100};
    }

    int s = entropy_buf.size();
    float ma;

    insert_buf(entropy_buf, cur_entropy);

    if (s < lambda){
        update_sum(cur_entropy);
        return {100, 0, 0, 0};
    } else if (s == lambda){
        ma = init_mean(cur_entropy);
        aggrM = ma;
        num_data += num_data;
        return {ma, 0, 0, 0};
    }

    float old_ma = running_mean;
    float old_sd = running_sd;
    ma = moving_avg(entropy_buf,lambda);
    running_sd = welford_var(ma, num_data);

    int cur_state = update_state(ma, old_ma, old_sd);
    int cur_mode = select_mode(cur_state);
    
    constraint_buf(entropy_buf);
    num_data += num_data;

    return {ma, running_sd, cur_state, cur_mode};
}

std::vector<float> Uncertainty::normalise(std::vector<float> &cp)
{	
    float mx = *max_element(std::begin(cp), std::end(cp));
    
    for(auto &elem : cp)
        elem = (float)elem / mx;
    
    return cp;
}

std::vector<float> Uncertainty::softmax(std::vector<float> &arg_vec)
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

    // cout<< "Probability list" <<endl;
    // print_vector(norm_vec);

    return norm_vec;
}

float Uncertainty::entropy(std::vector<float> &arg_vec)
{
    float sum = 0;
    for(auto const &elem : arg_vec){
        sum += elem * std::log2(1/elem);
    }
    //cout << "uncertainty: " << sum <<endl;
    return sum;
}

void Uncertainty::insert_buf(std::vector<float> &arg_vec, float &elem){
    arg_vec.insert (arg_vec.begin(), elem);

}

void Uncertainty::constraint_buf(std::vector<float> &arg_vec){
    //arg_vec.insert (arg_vec.begin(), elem);
    //arg_vec = normalise(arg_vec);
    if (arg_vec.size() > lambda){
        arg_vec.pop_back();
    }
}

void Uncertainty::update_sum(float &elem){
    data_sum += elem;
}

float Uncertainty::init_mean(int n){
    running_mean  = data_sum / n;
    return running_mean;
}


float Uncertainty::moving_avg(std::vector<float> &arg_vec, int n){

    float mean = running_mean - (arg_vec.back()/n) + arg_vec[0]/n;
    return mean;
}


float Uncertainty::welford_var(float x, int k){
    float oldM = aggrM;
    float newM = aggrM;
    float newS = aggrS;

    newM = newM + (x - newM)/k;
    newS = newS + (x-newM)*(x - oldM);

    aggrM = newM;
    aggrS = newS;

    float newsd = sqrt(newS/(k-1));

    return newsd;
}

int Uncertainty::update_state(float ma, float old_ma, float old_sd){
    if (ma >= (old_ma-old_sd) || ma <= (old_ma+old_sd)){
        if (state != 18){
            state += state;
            return state;
        }
    } else {
        state = 1;
        return state;
    }
}

int Uncertainty::select_mode(int n){
    if (state <= n){
        return 1;
    }else if (state <= (2*n + 1) ){
        return 2;
    }else if (state <= (3*n + 3)){
        return 3;
    }else if (state <= (4*n + 6)){
        return 4;
    }

}


//helper func: for printing vector
void Uncertainty::print_vector(std::vector<float> &vec)
{
	std::cout << "{ ";
	for(auto const &elem : vec)
	{
		std::cout << elem << " ";
	}
	std::cout << "}" <<endl;
}