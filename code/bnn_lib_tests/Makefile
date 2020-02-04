
# BNN

LIB_bnn = ./lib_bnn
LIB_hls = ./hls
SRC_DIR = ./src

CXX= g++

CFLAGS+=  -O2 -I . 
CFLAGS+= -Wno-unused-result -Wno-write-strings -Wno-deprecated-register  -std=c++14 -fopenmp
#CFLAGSA = $(CFLAGS) -std=c++11

#XI_CFLAGS = $(CFLAGS) -DXILINX -DOFFLOAD --std=gnu++11 -march=armv7-a -I $(LIB_bnn)  -I $(LIB_hls) -I ${CURDIR} -mfloat-abi=hard 

XI_CFLAGS = $(CFLAGS) -DXILINX -DOFFLOAD  -march=armv7-a -I $(LIB_bnn)  -I $(LIB_hls) -I $(SRC_DIR) -mfloat-abi=hard 

XI_CFLAGS+= -DNEON -mfpu=neon -funsafe-math-optimizations -ftree-vectorize -mvectorize-with-neon-quad -ftree-vectorizer-verbose=2 #If NEON SIMD instructions are to be used


XI_LDFLAGS = -L $(LIB_bnn)

LDFLAGS = -pthread -lpthread -z muldefs

XI_LDFLAGS+= -lrt -lkernelbnn 

XI_PROGs= BNN

SOURCE= $(SRC_DIR)/main.cpp   $(SRC_DIR)/kernelbnn.h

# OpenCV variables
OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

.PHONY: all clean odroid

all: odroid

odroid: $(XI_PROGs)

foldedmv-offload.o: $(SRC_DIR)/foldedmv-offload.cpp $(SRC_DIR)/foldedmv-offload.h
	$(CXX) -c $(SRC_DIR)/foldedmv-offload.cpp $(XI_CFLAGS)

rawhls-offload.o: $(SRC_DIR)/rawhls-offload.cpp 
	$(CXX) -c $(SRC_DIR)/rawhls-offload.cpp $(XI_CFLAGS)

win.o: $(SRC_DIR)/win.cpp $(SRC_DIR)/win.hpp
	$(CXX) -c $(SRC_DIR)/win.cpp -I $(SRC_DIR) -std=c++14

roi_filter.o: $(SRC_DIR)/roi_filter.cpp $(SRC_DIR)/roi_filter.hpp
	$(CXX) -c $(SRC_DIR)/roi_filter.cpp $(LIBS) -std=c++14 -fopenmp -DXILINX -DOFFLOAD  -march=armv7-a -I $(SRC_DIR) -mfloat-abi=hard 

BNN: $(SOURCE) foldedmv-offload.o rawhls-offload.o win.o roi_filter.o
	$(CXX) -o $@ $< foldedmv-offload.o rawhls-offload.o win.o roi_filter.o $(LIBS) $(XI_CFLAGS) $(XI_LDFLAGS)  $(LDFLAGS)

clean-all:
	rm -f  $(XI_PROGs) foldedmv-offload.o rawhls-offload.o win.o roi_filter.o

clean:
	rm -f  $(XI_PROGs) roi_filter.o