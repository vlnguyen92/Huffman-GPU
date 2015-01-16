#ifndef HIST_H
#define HIST_H

__global__ void histo_kernel(unsigned char* buffer, long size, unsigned int* histo);

int run(char* file, unsigned int* freq);
#endif
