#ifndef win
#define win
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

class Win_filter{
    private:
        void print_vector(std::vector<float> vec);
        float expDecay(float lambda, int t, int N = 1); //supporting math functions
        //std::vector<float> normalise(std::vector<float> &vec); //supporting math functions
        vector<float> calculate_certainty(const std::vector<float> &arg_vec); //supporting math functions

    public:
        unsigned int wstep;
        unsigned int wlength;
        std::vector<float> wweights;
        std::vector<std::vector<float>> wmemory;
        unsigned int wpast_output;
        unsigned int wcount;

        Win_filter(unsigned int step, unsigned int length){
            wstep = step;
            wlength = length;
            wweights.resize(length);
            wcount = 0;
        }

        unsigned int analysis();
        void init_weights(float lambda);
        void update_memory(const std::vector<float> &class_result);
};

#endif
