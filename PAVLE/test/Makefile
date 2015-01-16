NVCC=nvcc
CC = g++

CUDA_INCLUDEPATH=/usr/local/cuda-6.5/include

NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64

GCC_OPTS=-O3 -Wall -Wextra -m64

OBJ = main_test_cu.o #vlc_kernel_sm32.o scan.o pack_kernels.o stats_logger.cpp 
#HDRS = cuda_helpers.h load_data.h print_helpers.h stats_logger.h stdafx.h 
SRC = main_test_cu.cpp #stats_logger.cpp 
EXC = main

main: $(SRC)
	$(CC) $(SRC) -o $(EXC)
	
#-o VLEtest main.o student_func.o compare.o reference_calc.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(NVCC_OPTS)

#main.o: main.cpp timer.h utils.h reference_calc.cpp compare.cpp HW1.cpp
#	g++ -c main_test_cu.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) -I $(OPENCV_INCLUDEPATH)
#
#scan.o: scan.cu utils.h
#	nvcc -c scan.cu $(NVCC_OPTS)
#
#compare.o: compare.cpp compare.h
#	g++ -c compare.cpp -I $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)
#
#reference_calc.o: reference_calc.cpp reference_calc.h
#	g++ -c reference_calc.cpp -I  $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)
#
clean:
	rm -f *.o *.png $(EXC) 
