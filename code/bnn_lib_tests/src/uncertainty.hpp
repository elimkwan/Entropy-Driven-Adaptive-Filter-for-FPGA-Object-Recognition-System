/******************************************************************************
 * Code developed by Elim Kwan in April 2020 
 *
 * Uncertainty Estimation (Uncertainty Filter)
 * Calculate uncertainty in BNN output with: Entropy, Variance, AutoCorrelation 
 * 
 *****************************************************************************/
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

        int __lambda;
        //int __state;
        bool __init;
        std::vector<double> __uncertainty_score_buf;
        std::vector<double> __uncertainty_score_runningmean_buf;
        double __uncertainty_score_sum;
        double __uncertainty_score_running_mean;
        double __sd_of_uncertainty_score_running_mean;
        double __mean_of_uncertainty_score_runningmean;
        double __aggrM;
        int __count;
        double __alpha; 
        std::vector<std::vector<double>> __corr_history;
        bool __corr_init = false;
        std::vector<double> __running_corr;

        //Calculate Variance, Entropy, AutoCorrelation
        double cal_entropy(std::vector<double> &arg_vec);
        vector<double> cal_variance(vector<double> &ma, int n);
        vector<double> running_var(vector<double> ma, int n, double mean_of_ma, double arg_aggrM);
        double cal_autocorr(std::vector<double> &arg_vec, int result);
        int insert_corr_history(std::vector<double> &arg_vec);
        void constraint_corr_history(int n);
        double running_autocorr(std::vector<double> &arg_vec, double old_corr);
        double running_autocorr_init(std::vector<double> &arg_vec);


        std::vector<double> normalise(std::vector<double> &cp);
        std::vector<double> softmax(std::vector<double> &arg_vec);
        void insert_buf(std::vector<double> &arg_vec, double &elem);
        void constraint_buf(std::vector<double> &arg_vec);
        double running_mean_init(double &elem);
        double running_mean(std::vector<double> &arg_vec, int n);
        double naive_mean(std::vector<double> &arg_vec, int n);

        int ps_mode(bool initialised, double new_sd);
        //int update_state(double sample, double old_ma, double old_sd, float alpha, int n);
        //int select_powersaving_mode(int n);

        void print_vector(std::vector<double> &vec);
        //void set_dataSum(double &elem);

    public:
        
        Uncertainty(int lamda){
            __lambda = lamda;

            //__state = 0;
            __init = false;
            __uncertainty_score_sum = 0;
            __uncertainty_score_running_mean = 0;
            __sd_of_uncertainty_score_running_mean = 0;
            __aggrM = 0;
            __mean_of_uncertainty_score_runningmean = 0;
            __count = 0;
            __alpha = 0;
            __corr_init = false;
        }

        std::vector<double> cal_uncertainty(std::vector<float> class_result, string mode, int result);

};

#endif
