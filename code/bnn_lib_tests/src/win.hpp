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
        //std::vector<float> normalise(std::vector<float> &vec); //supporting math functions
        vector<float> calculate_certainty(const std::vector<float> &arg_vec); //supporting math functions
        vector<int> select_ws_wl(int mode);

    public:
        unsigned int wstep;
        unsigned int wlength;
        unsigned int max_wlength;
        std::vector<float> wweights;
        std::vector<std::vector<float>> wmemory;
        unsigned int wpast_output;
        unsigned int wcount;

        Win_filter(float temp, int step, int length){
            wstep = step;
            wlength = length;
            max_wlength = length;
            wweights.resize(length);
            wpast_output = 0;
            wcount = 0;
        }

        unsigned int analysis(int mode, bool flex);
        void init_weights(float lambda);
        void update_memory(const std::vector<float> &class_result);
};

#endif
