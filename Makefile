CXX				:= g++
NVCC			:= nvcc

CUDA_PATH		?= /usr/local/cuda

ifeq ($(shell uname -m), x86_64)
  LIBARCH=lib64
else
  LIBARCH=lib
endif

BIN_DIR			:= bin
SRC_DIR			:= src

CLFFT_PATH		:= /usr/local

INC_DIRS		:= include

CPU_LIBS		:= fftw3f

CUDA_LIBS		:= cufft
CUDAINC_DIRS	:= $(CUDA_PATH)/include

CL_LIBS			:= OpenCL clFFT
CLINC_DIRS		:= $(CLFFT_PATH)/include
CLLIB_DIRS		:= $(CLFFT_PATH)/lib64

CPU_FLAGS		:= $(patsubst %,-l%, $(CPU_LIBS))
CPU_FLAGS		+= $(patsubst %,-I%, $(INC_DIRS))

CUDA_FLAGS		:= $(patsubst %,-l%, $(CUDA_LIBS))
CUDA_FLAGS		+= $(patsubst %,-I%, $(INC_DIRS))
CUDA_FLAGS		+= $(patsubst %,-I%, $(CUDAINC_DIRS))

CL_FLAGS		:= $(patsubst %,-l%, $(CL_LIBS))
CL_FLAGS		+= $(patsubst %,-I%, $(INC_DIRS))
CL_FLAGS		+= $(patsubst %,-I%, $(CLINC_DIRS))
CL_FLAGS		+= $(patsubst %,-L%, $(CLLIB_DIRS))

all: main-build

cpu: pre-build
	$(CXX) -std=c++11 $(CPU_FLAGS) $(SRC_DIR)/fft_benchmark_cpu.cpp $(SRC_DIR)/common.cpp -o $(BIN_DIR)/fft_benchmark_cpu

cuda: pre-build
	$(NVCC) $(CUDA_FLAGS) $(SRC_DIR)/fft_benchmark_cuda.cu $(SRC_DIR)/common.cpp -o $(BIN_DIR)/fft_benchmark_cuda

opencl: pre-build
	$(CXX) -std=c++11 $(CL_FLAGS) $(SRC_DIR)/fft_benchmark_opencl.cpp $(SRC_DIR)/common.cpp -o $(BIN_DIR)/fft_benchmark_opencl

clean:
	@rm -rf $(BIN_DIR)

pre-build:
	@mkdir -p $(BIN_DIR)

main-build: pre-build
	@$(MAKE) --no-print-directory cpu cuda opencl
