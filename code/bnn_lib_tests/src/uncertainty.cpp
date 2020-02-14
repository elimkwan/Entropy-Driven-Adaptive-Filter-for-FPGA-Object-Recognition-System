#include "uncertainty.hpp"

std::vector<float> Uncertainty::wrapper(std::vector<float> class_result, int roi_area, int nn_output){

    std::vector<float> pmf = softmax(class_result);
    float cur_entropy = entropy(pmf);
    float cur_cross_entropy = cross_entropy(pmf,past_pmf);

    if (isnan(cur_entropy) || isnan(cur_cross_entropy)){
        cout << "entropy is nan" << endl;
        return {100.f, 0.f, 0.f,0.f, 0.f, 0.f};
    }

    update_buf(entropy_buf, cur_entropy);
    update_buf(cross_entropy_buf, cur_cross_entropy);

    std::cout << "---debug 2---- " << endl;
    float ma = moving_avg(entropy_buf,lambda);
    update_buf(ma_buf, ma);

    std::cout << "---debug 3---- " << endl;
    float d_ma = diff(ma_buf, lambda);

    std::cout << "---debug 4---- " << endl;
    float ma_cross = moving_avg(cross_entropy_buf, lambda);

    std::cout << "---debug 5---- " << endl;
    float a = roi_area;
    update_buf(roi_area_buf, a);
    std::cout << "---debug 6---- " << endl;
    float d_area = diff(roi_area_buf, lambda);

    std::cout << "---debug 7---- " << endl;
    float d_out = diff_out(nn_output, past_out);

    float f1 = h0*ma + h1*abs(d_ma) + h2*ma_cross + h3*abs(d_area) + h4*d_out;

    //update class member
    past_out = nn_output;
    past_pmf = pmf;

    return {f1, h0*ma, h1*abs(d_ma), h2*ma_cross, h3*abs(d_area), h4*d_out};

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

float Uncertainty::cross_entropy(std::vector<float> &p1, std::vector<float> &p2)
{
    float sum = 0;
    int i = 0;
    for(auto const &elem : p1){
        sum += elem * log2(elem/p2[i]);
        i++;
    }
    //cout << "uncertainty: " << sum <<endl;
    return sum;
}

float Uncertainty::sd(std::vector<float> &arg_vec)
{	
    float mean = 0;
    int size = arg_vec.size();
    for(auto const &elem : arg_vec){
        mean += elem;
    }
    mean = mean/size;

    float sum = 0;
    for(auto const &elem : arg_vec){
        sum += pow((elem-mean),2);
    }
    return sqrt(sum/size);
}

void Uncertainty::update_buf(std::vector<float> &arg_vec, float &elem){
    arg_vec.insert (arg_vec.begin(), elem);

    arg_vec = normalise(arg_vec);

    if (arg_vec.size() > 10){
        arg_vec.pop_back();
    }
}

float Uncertainty::moving_avg(std::vector<float> &arg_vec, int n){
    float sum = 0.f;
    cout<<"moving average: "<<endl;
    print_vector(arg_vec);
    if (arg_vec.size() < n){
        cout<<"less than: n and size: " << n << " "<< arg_vec.size() <<endl;
        //n = arg_vec.size();
        sum = accumulate(arg_vec.begin(), arg_vec.end(), sum);
        return (sum/n);
    }

    cout<<"n and size: " << n << " "<< arg_vec.size() <<endl;

    for (int i = 0; i < n; i++) {
        sum += arg_vec[i];
    }

    return (sum/n);

    // std::vector<float> v;
    // std::copy(arg_vec.begin(), arg_vec.begin()+(n-1), v.begin());
    // cout<<"summing"<<endl;
    // sum = accumulate(v.begin(), v.end(), sum);

    // cout<<"sum and sum/n" << sum  << " " << sum/n <<endl;
    // return (sum/n);
    
}

float Uncertainty::diff(std::vector<float> &arg_vec, int n){
    int s = arg_vec.size();
    float sum = 0.f;
    if (s < 2){
        return 0.0;
    } else if(s < n){
        cout<<"diff less than: n and size: " << n << " "<< arg_vec.size() <<endl;
        //n = arg_vec.size();
        sum = accumulate(arg_vec.begin(), arg_vec.end(), sum);
        float ans = arg_vec[0] - (sum-arg_vec[0])/s;
        return ans;

    } else {
        // std::vector<float> v;
        // copy(arg_vec.begin(), arg_vec.begin()+(n-2), v.begin());
        // float sum = 0.0;;
        // sum = accumulate(v.begin(), v.end(), sum);
        // return (arg_vec[0] - sum/(n-1));
        for (int i = 1; i < n; i++) {
            sum += arg_vec[i];
        }

        return (arg_vec[0]-sum/(n-1));

    }
}

float Uncertainty::diff_out(int cur, int past){
    if (cur == past){
        return 0.0; //dout/dt =0
    } else {
        return 1.0;
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