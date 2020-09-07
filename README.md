# RealTimeObjectRecognition
This program is for Real Time Object Recognition with Binary Neural Network on FPGA. Our main contribution is appending an Entropy driven Adaptive Filter for a more accurate and resources efficient system. This includes:
- Entropy-based Uncertainty Estimation {*uncertainty.cpp*}
- Window Filter with variable step size and length {*win.cpp*} which changes dynamically based on level of uncertainty in data. Enabling us to decimate frames as well
- Region-Of-Interest(ROI) Detection with Optical Flow and Contour Detection {*roi_filter.cpp*}
- Other possible power-saving features by altering configurations in {*main.cpp*, *main-adaptivefil.cpp*, *main-uncertainty.cpp*, *main-windowfil.cpp*}
    * Dynamic clock
    * ROI Filter - alternate between Optical Flow, Contour Detection and Reuse Past ROI based on level of uncertainty in data


## Folder Architecture:
- *code/bnn_lib_tests/Makefile* is the Makefile
- *code/bnn_lib_tests/src/\** contains all the active sources files
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
- Then in the bnnlibtests folder, run “make clean all"
Upon compiling the program with the Makefile, 4 executables can be called: BNN, WindowFilExp,UncertaintyExp, AdaptiveFilExp.

---
## Case 1: With Webcam Input


Uses webcam input for classification. Classify it as one of the ten classes ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

Command avaliable:
./BNN [No. of frame] [Schemes] [Expected Class]
[No. of frame]: number of frames to be Captured
[Schemes]: either A/B/C/base, adaptive filtering schemes to be applied, with A being most accurate and C most resources efficient
[Expected Class]: Enter the expected classification results, for analysing the system accuracy.

Example 1: 
``` 
./BNN 500 A 4
```
The command means capture 500 frames, and apply scheme A for adaptive filter, expecting the input to be deer

Example 2: 
``` 
./BNN 500 base 1
```
The command means capture 500 frames, and apply base model for adaptive filter, expecting the input to be automobile


## Case 2: With Video Dataset as Input

Experiment for analysing the performance of different adaptive filter schemes under different scenarios.
Scheme A is optimised for accuracy.
Scheme B is optimised for accuracy and efficiency.
Scheme C is optimised for accuracy and efficiency with more aggressive computational savings.
Dataset is used instead of the webcam.
Dataset Directory: ../experiments/DatasetX (X ranges from 1 - 5)
Output Log Directory: ../experiments/result/result-overview.csv

For Scheme A :
```
./AdaptiveFilExp 1 1 10 15 12 15 1 10 10 13 
```
For Scheme B :
```
./AdaptiveFilExp 1 1 15 15 15 12 15 10 10 8
```
For Scheme C :
```
./AdaptiveFilExp 1 1 10 8 15 12 15 10 10 6
```

Users can also self-specified different scheme: ./BNN SS-1 WL-1 SS-2 WL-2 SS-3 WL-3 SS-4 WL-4 SS-5 WL-5 (Replace the numbers with your own StepSize and WindowLength Sets)

## Case 3: Compare Uncertainty Estimation Schemes Experiments

Experiment for analysing the performance of different uncertainty estimation schemes under different scenarios. Dataset is used instead of the webcam. There are three estimation schemes:
- Entropy
- Autocorrelation
- Variance

Dataset Directory: ../experiments/uncertainty-datasetX (X ranges from 2 - 5)
Output Log Directory: ../experiments/result/result-overview.csv

Run the experiment with the following command:
```
./UncertaintyExp
```


## Case 4: Optimise Window Filter Configurations Experiments

Experiment for finding the optimium window filter settings (Window Step Size & Window Length) for different scenarios. Dataset is used instead of the webcam.

Dataset Directory: ../experiments/datasetX (X ranges from 1 - 5)
Output Log Directory: ../experiments/result/result-overview.csv

Run the experiment with the following command:
```
./WindowFilExp
```

## Other options
Region-of-Interst code is also embedded in the file. Users can change the roi_config in the main files from "full-roi" to "opt-roi", "cont-roi","eff-roi", which correspond to optical flow detection, contour detection and hybrid of the two
