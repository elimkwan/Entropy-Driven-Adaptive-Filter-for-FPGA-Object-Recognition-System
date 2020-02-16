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

        std::vector<float> normalise(std::vector<float> &cp);
        std::vector<float> softmax(std::vector<float> &arg_vec);
        float entropy(std::vector<float> &arg_vec);
        void insert_buf(std::vector<float> &arg_vec, float &elem);
        void constraint_buf(std::vector<float> &arg_vec);
        void update_sum(float &elem);
        float init_mean(int n);
        float moving_avg(std::vector<float> &arg_vec, int n);
        float welford_var(float x, int k);
        int update_state(float ma, float old_ma, float old_sd);
        int select_mode(int n);

    public:

        std::vector<float> entropy_buf;
        std::vector<float> ma_buf;
        float h0, h1, h2, h3, h4;
        int lambda;
        int state;
        float data_sum;
        float running_mean;
        float running_sd;
        int num_data;
        float aggrM;
        float aggrS;
        
        Uncertainty(int arg_lamda){
            lambda = arg_lamda;
            state = 0;
            num_data = 0;
            aggrS = 0;
        }

        std::vector<float> wrapper(std::vector<float> class_result);

};

#endif
