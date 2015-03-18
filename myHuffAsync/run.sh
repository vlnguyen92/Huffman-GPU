nvcc -arch=sm_20 -c hist.cu -o hist.o
nvcc -g -G huffman.cu hist.o -o huff
./huff test1024*.in 
