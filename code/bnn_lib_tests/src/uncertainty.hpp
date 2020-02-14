#ifndef uncertainty
#define uncertainty
#include <iostream>
#include <numeric>
#include <math.h>
#include <vector>
#include <algorithm>
#include <iterator>

using namespace std;

class Uncertainty{
    private:
        void print_vector(std::vector<float> &vec);

    public:

        std::vector<float> entropy_buf;
        std::vector<float> cross_entropy_buf;
        std::vector<float> ma_buf;
        std::vector<float> roi_area_buf;
        float past_out;
        //std::vector<float> raw_output_buf;
        std::vector<float> past_pmf;
        float h0, h1, h2, h3, h4;
        int lambda;
        
        Uncertainty(float arg_h0, float arg_h1, float arg_h2, float arg_h3, float arg_h4, int arg_lamda){
            h0 = arg_h0;
            h1 = arg_h1;
            h2 = arg_h2;
            h3 = arg_h3;
            h4 = arg_h4;
            lambda = arg_lamda;

            past_pmf = {0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f};
        }

        std::vector<float> wrapper(std::vector<float> class_result, int roi_area, int nn_output);

        std::vector<float> normalise(std::vector<float> &cp);
        std::vector<float> softmax(std::vector<float> &arg_vec);
        float entropy(std::vector<float> &arg_vec);
        float cross_entropy(std::vector<float> &p1, std::vector<float> &p2);
        float sd(std::vector<float> &arg_vec);
        void update_buf(std::vector<float> &arg_vec, float &elem);
        float moving_avg(std::vector<float> &arg_vec, int n);
        float diff(std::vector<float> &arg_vec, int n);
        float diff_out(int cur, int past);
};

#endif
