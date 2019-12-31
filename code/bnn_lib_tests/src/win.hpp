#ifndef win
#define win

class Win_filter{
    public:
        unsigned int step;
        unsigned int length;
        unsigned int past_output;
        std::vector<std::vector<float>> memory;
        std::vector<float> weights;

        Win_filter(){
            step = 0;
            length = 0;
        }

        void analysis();
    
};

#endif
