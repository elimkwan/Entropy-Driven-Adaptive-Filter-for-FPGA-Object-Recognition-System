# RealTimeObjectRecognition
Work Progress is documented at: https://gist.github.com/elimkwan/291b4fa29cfe936bcdbcf21526b5c6a8


This program is for Real Time Object Recognition with Binary Neural Network on FPGA. Our main contribution is appending pre and post processing modules to customise the system to be more adaptive, this includes:
- Entropy-based Uncertainty Estimation {*uncertainty.cpp*}
- Window Filter with variable step size and length {*win.cpp*}
- Region-Of-Interest(ROI) Detection with Optical Flow and Contour Detection {*roi_filter.cpp*}
- Power-Saving Features {*main.cpp*}
    * Decimate frames
    * Dynamic clock
    * Flexible Window Filter - alternate between varies step size based on level of uncertainty in data
    * ROI Filter - alternate between Optical Flow, Contour Detection and Reuse Past ROI based on level of uncertainty in data


## Folder Architecture:
- *code/bnn_lib_tests/Makefile* is the Makefile
- *code/bnn_lib_tests/src/\** contains all the active sources files
- *code/bnn_lib_tests/all-main-cpp/\** contains all the cpp scripts for testing
- *code/bnn_lib_tests/experiments/results* is the directory for active test results
- *code/bnn_lib_tests/experiments/images* is the directory for active dataset
- *testing/data/* contains all 12 datasets
- *testing/TrailX* contins results of previous tests


## Dependencies:
- Avnet Zedboard with Linux GCC 5.2.1
- libkernelbnn.a file:  the compiled BNN (includekernelbnn.hin headers to use it)
- sdslib.h file:  part of SDSoC environment API, which provides functions to map memory spaces
- foldmv-offload files:  for managing hardware offload•rawhls-offload files:  for execution of HLS souce code
- OpenCV 2.4.9 Library:  for image processing
- OpenMP 4.0 Library:  for pipeline functions

---
## Setting up the FPGA
After wiring up the FPGA to a linux PC, execute the program by following the steps below:
- To access the files, in the File system of the laptop, choose other location, and type:
```
sftp://198.168.137.99
```

- SSH to the remote sever at the terminal and initialise the FPGA via flashing the neural networkto it:
```
ssh −X xilinx@192.168.137.99
password: xilinx
/home/xilinx/: su
/home/xilinx/: cp .Xauthority /root/
/home/xilinx/jose_bnn/: echo kernelbnn.linked.bit.bin > /sys/class/fpga_manager/fpga0/firmware
```
- Then in the bnnlibtests folder, run “make clean” or “make clean-all”, make clean will allow a faster, less than 60-second build.  Then, run “make”.

---
## Example 1: With Webcam Input

Upon compiling the program with the Makefile, a BNN executable can be called with following arguements:
```
./BNN [No. of Frame] [Uncertainty Scheme] [Window Filter Scheme] [ROI Filter Scheme] [Dynmic Clock] [Base Case] [Expected Class]
```

Command for proposed scheme:
The program will process 500 frames, with entropy-based analysis, dropping frames, uses flexible window filter, uses flexible ROI filter, uses dynamic clock and not a based case, with expected class equals to 4
```
./BNN 500 en drop flexw eff-roi dynclk nbase 4
```

Other combinations:
```
./BNN 500 na notdrop notflexw full-roi ndynclk base 4
./BNN 500 en notdrop notflexw full-roi ndynclk nbase 4
./BNN 500 en notdrop notflexw opt-roi ndynclk nbase 4
./BNN 500 en notdrop notflexw cont-roi ndynclk nbase 4
./BNN 500 en notdrop notflexw eff-roi ndynclk nbase 4
./BNN 500 en notdrop flexw eff-roi ndynclk nbase 4


./BNN 500 en drop notflexw full-roi ndynclk nbase 4
./BNN 500 en drop notflexw opt-roi ndynclk nbase 4
./BNN 500 en drop notflexw cont-roi ndynclk nbase 4
./BNN 500 en drop notflexw eff-roi ndynclk nbase 4
./BNN 500 en drop flexw eff-roi ndynclk nbase 4
```

---
## Example 2: With Datasets

- Copy the dataset to be used to *code/bnn_lib_tests/experiments/images*
- Depends on the test to be carried out, replace *code/bnn_lib_tests/src/main.cpp* with one of the testing scripts in *code/bnn_lib_tests/all-main-cpp/*
- Change the test setting in the scripts
- Recompile and run the program with:
```
./BNN
```
