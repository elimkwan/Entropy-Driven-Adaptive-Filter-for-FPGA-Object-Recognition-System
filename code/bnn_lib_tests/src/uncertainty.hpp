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

        std::vector<double> __entropy_buf;
        std::vector<double> __ma_buf;
        int __lambda;
        int __state;
        double __data_sum;
        double __running_mean;
        double __running_sd;
        double __aggrM;
        double __mean_of_ma;
        int __count;

        // std::vector<double> __class0_buf;
        // std::vector<double> __class1_buf;
        // std::vector<double> __class2_buf;
        // std::vector<double> __class3_buf;
        // std::vector<double> __class4_buf;
        // std::vector<double> __class5_buf;
        // std::vector<double> __class6_buf;
        // std::vector<double> __class7_buf;
        // std::vector<double> __class8_buf;
        // std::vector<double> __class9_buf;
        // std::vector<double> __c_dataSum;
        // std::vector<double> __c_rolling_mean;
        // std::vector<double> __c_rolling_sd;


        void print_vector(std::vector<double> &vec);
        std::vector<double> normalise(std::vector<double> &cp);
        std::vector<double> softmax(std::vector<double> &arg_vec);
        double entropy(std::vector<double> &arg_vec);
        void insert_buf(std::vector<double> &arg_vec, double &elem);
        void constraint_buf(std::vector<double> &arg_vec);
        void set_dataSum(double &elem);
        double init_running_mean(double &elem);
        double moving_avg(std::vector<double> &arg_vec, int n);
        double naive_avg(std::vector<double> &arg_vec, int n);
        vector<double> moving_var(vector<double> ma, int n, double mean_of_ma, double arg_aggrM);
        int update_state(double sample, double old_ma, double old_sd, float alpha);
        int select_mode(int n);
        vector<double> init_var(vector<double> &ma, int n);
        double __alpha; 

    public:
        
        Uncertainty(int arg_lamda){
            __lambda = arg_lamda;

            __state = 0;
            __data_sum = 0;
            __running_mean = 0;
            __running_sd = 0;
            __aggrM = 0;
            __mean_of_ma = 0;
            __count = 0;
            __alpha = 0;
        }

        std::vector<double> cal_uncertainty(std::vector<float> class_result, string mode);

};

#endif
