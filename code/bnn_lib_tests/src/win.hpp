/******************************************************************************
 * Code developed by Elim Kwan in April 2020 
 *
 * Window Filter
 * Generate aggregates of classification results.
 * Step Size and Length of Filter can be varied.
 * 
 *****************************************************************************/
#ifndef win
#define win
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

class Win_filter{
    private:
        void print_vector(std::vector<float> &vec);
        float expDecay(float lambda, int t, int N = 1); //supporting math functions
        vector<float> calculate_softmax(const std::vector<float> &arg_vec); //supporting math functions
        vector<int> select_ws_wl(int mode, int aa, int bb, int cc, int dd, int ee, int ff, int gg, int hh, int ii, int jj);
        void update_memory(const std::vector<float> &class_result);
        int display_c;
        bool display_f;

    public:
        unsigned int wstep;
        unsigned int wlength;
        unsigned int max_wlength;
        std::vector<float> wweights;
        std::vector<std::vector<float>> wmemory;
        unsigned int wpast_output;
        unsigned int wcount;
        bool winit;

        Win_filter(int step, int length){
            wstep = step;
            wlength = length;
            max_wlength = 20;
            wweights.resize(20);
            wpast_output = 0;
            wcount = 0;
            display_c = 0;
            display_f = false;
            bool winit = false;
        }

        unsigned int analysis(const std::vector<float> &class_result, int mode, bool flex, int aa, int bb, int cc, int dd, int ee, int ff, int gg, int hh, int ii, int jj);
        void init_weights(float lambda);
        bool dropf();
        bool processf();
        bool get_display_f();
};

#endif
