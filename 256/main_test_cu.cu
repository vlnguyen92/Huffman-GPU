/*
 * PAVLE - Parallel Variable-Length Encoder for CUDA. Main file.
 *
 * Copyright (C) 2009 Ana Balevic <ana.balevic@gmail.com>
 * All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify it under the terms of the
 * MIT License. Read the full licence: http://www.opensource.org/licenses/mit-license.php
 *
 * If you find this program useful, please contact me and reference PAVLE home page in your work.
 * 
 */

#include "stdafx.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include "cuda_helpers.h"
#include "uint256.h"
#include "print_helpers.h"
#include "comparison_helpers.h"
#include "stats_logger.h"
#include "load_data.h"
//#include "cutil.h"

#include "vlc_kernel_sm64huff.cu"
#include "vlc_kernel_sm32.cu"
#include "scan.cu"
#include "pack_kernels.cu"
#include "cpuencode.cpp"

void runVLCTest(char *file_name, uint num_block_threads, uint num_blocks=1);

extern "C" void cpu_vlc_encode(unsigned int* indata, unsigned int num_elements, unsigned int* outdata, unsigned int *outsize, uint256 *codewords, unsigned int* codewordlens);
extern "C" void cpu_vlc_encode_lame(unsigned int* sourceData, unsigned int num_elements, 
					unsigned int* destData, unsigned int *outsize, 
					unsigned int *codewords, unsigned int* codewordlens); 

int main(int argc, char* argv[]){
	if(!InitCUDA()) { return 0;	}
	unsigned int num_block_threads = 256;
	if (argc > 1)
		for (int i=1; i<argc; i++)
			runVLCTest(argv[i], num_block_threads);
	else {runVLCTest(NULL, num_block_threads, 1024);	}
	CUDA_SAFE_CALL(cudaThreadExit());
	return 0;
}

//void runVLCTest(unsigned int num_blocks, unsigned int num_block_threads, unsigned int nnsymbols) { 
void runVLCTest(char *file_name, uint num_block_threads, uint num_blocks) {
	printf("CUDA! Starting VLC Tests!\n");
	unsigned int num_elements; //uint num_elements = num_blocks * num_block_threads; 
	unsigned int mem_size; //uint mem_size = num_elements * sizeof(int); 
	unsigned int symbol_type_size = sizeof(int);
	//////// LOAD DATA ///////////////
	double H; // entropy
	initParams(file_name, num_block_threads, num_blocks, num_elements, mem_size, symbol_type_size);
	printf("Parameters: num_elements: %d, num_blocks: %d, num_block_threads: %d\n----------------------------\n", num_elements, num_blocks, num_block_threads);
  printf("input size: %d\n", mem_size);
	////////LOAD DATA ///////////////
	uint	*sourceData =	(uint*) malloc(mem_size);
	uint	*destData   =	(uint*) malloc(mem_size);
	uint	*crefData   =	(uint*) malloc(mem_size*16);

	uint256	*codewords	   = (uint256*) malloc(NUM_SYMBOLS*sizeof(uint256));
	uint	*codewordlens  = (uint*) malloc(NUM_SYMBOLS*symbol_type_size);

	uint	*cw32 =		(uint*) malloc(mem_size);
	uint	*cw32len =	(uint*) malloc(mem_size);
	uint	*cw32idx =	(uint*) malloc(mem_size);

	uint	*cindex2=	(uint*) malloc(num_blocks*sizeof(int));

	memset(sourceData,   0, mem_size);
	memset(destData,     0, mem_size);
	memset(crefData,     0, mem_size);
	memset(cw32,         0, mem_size);
	memset(cw32len,      0, mem_size);
	memset(cw32idx,      0, mem_size);
	memset(codewords,    0, NUM_SYMBOLS*symbol_type_size);
	memset(codewordlens, 0, NUM_SYMBOLS*symbol_type_size);
	memset(cindex2, 0, num_blocks*sizeof(int));
	//////// LOAD DATA ///////////////
	loadData(file_name, sourceData, codewords, codewordlens, num_elements, mem_size, H);
	//printf("Codewords 32bit:\n");
	//print_array_in_hex(codewords, 256);
	//print_array<uint>(codewordlens, 256);
	//////// LOAD DATA ///////////////

	unsigned int	*d_sourceData, *d_destData, *d_destDataPacked;
	unsigned int	*d_codewordlens;
  uint256       *d_codewords;
	unsigned int	*d_cw32, *d_cw32len, *d_cw32idx, *d_cindex, *d_cindex2;

	CUDA_SAFE_CALL(cudaMalloc((void**) &d_sourceData,		  mem_size));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_destData,			  mem_size));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_destDataPacked,	  mem_size));

	CUDA_SAFE_CALL(cudaMalloc((void**) &d_codewords,		  NUM_SYMBOLS*sizeof(uint256)));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_codewordlens,		  NUM_SYMBOLS*symbol_type_size));

	CUDA_SAFE_CALL(cudaMalloc((void**) &d_cw32,				  mem_size));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_cw32len,			  mem_size));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_cw32idx,			  mem_size));

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_cindex,         num_blocks*sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_cindex2,        num_blocks*sizeof(unsigned int)));

	CUDA_SAFE_CALL(cudaMemcpy(d_sourceData,		sourceData,		mem_size,		cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_codewords,		codewords,		NUM_SYMBOLS*symbol_type_size,	cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_codewordlens,	codewordlens,	NUM_SYMBOLS*symbol_type_size,	cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_destData,		destData,		mem_size,		cudaMemcpyHostToDevice));

    dim3 grid_size(num_blocks,1,1);
    dim3 block_size(num_block_threads, 1, 1);
	// unsigned int sm_block_size_for_cwlens = num_block_threads * sizeof(unsigned int);
	unsigned int sm_size; 
//  printf("size: %d\n", sizeof(uint256));
	unsigned int NT = 10; //number of runs for each execution time
	float ktime = 0.0f;
	unsigned int timer = 0;
//	CUT_SAFE_CALL(cutCreateTimer(&timer));


	//////////////////* CPU ENCODER *///////////////////////////////////
	ktime = 0;
//	CUT_SAFE_CALL(cutResetTimer(timer));
//	CUT_SAFE_CALL(cutStartTimer(timer));
	unsigned int refbytesize;
	cpu_vlc_encode((unsigned int*)sourceData, num_elements, (unsigned int*)crefData,  &refbytesize, codewords, codewordlens);
//  print_array_ints_as_bits(codewords,NUM_SYMBOLS);
//  print_array_ints_as_bits(codewordlens,NUM_SYMBOLS);
	//cpu_vlc_encode_lame((unsigned int*)sourceData, num_elements, (unsigned int*)crefData,  &refbytesize, codewords, codewordlens);
//	CUT_SAFE_CALL(cutStopTimer(timer));
//	ktime = cutGetTimerValue(timer);
	printdbg_data_bin("input.txt", sourceData, num_elements); 
	printf("CPU Encoding time (CPU): %f (ms)\n", ktime);
	printf("CPU Encoded to %d [B]\n", refbytesize);
	unsigned int num_ints = refbytesize/4 + ((refbytesize%4 ==0)?0:1);
	printdbg_data_bin("cpuout.txt", crefData, num_ints);
	//////////////////* END CPU *///////////////////////////////////


#if 0
	//////////////////* SM32 KERNEL *///////////////////////////////////
    grid_size.x		= num_blocks;
    block_size.x	= num_block_threads;
	sm_size			= block_size.x*sizeof(unsigned int);
#ifdef CACHECWLUT
	sm_size			= 2*NUM_SYMBOLS*sizeof(int) + block_size.x*sizeof(unsigned int);
#endif
	ktime			= 0;
//	CUT_SAFE_CALL(cutResetTimer(timer));
//	CUT_SAFE_CALL(cutStartTimer(timer));
	for (int i=0; i<NT; i++) {
	vlc_encode_kernel_sm32<<<grid_size, block_size, sm_size>>>(d_sourceData, d_codewords, d_codewordlens,  
																					d_cw32, d_cw32len, d_cw32idx, 
																d_destData, d_cindex); //testedOK2
	}
	cudaThreadSynchronize();
	CUT_CHECK_ERROR("Kernel execution failed\n");
//	CUT_SAFE_CALL(cutStopTimer(timer));
//	ktime += cutGetTimerValue(timer);
	printf("GPU Encoding time (SM32): %f (ms)\n", ktime/NT);
	//////////////////* END KERNEL *///////////////////////////////////
#endif



#if 1
	//////////////////* SM64HUFF KERNEL *///////////////////////////////////
    grid_size.x		= num_blocks;
    block_size.x	= num_block_threads;
	sm_size			= block_size.x*sizeof(unsigned int);
#ifdef CACHECWLUT
	sm_size			= 2*NUM_SYMBOLS*sizeof(int) + block_size.x*sizeof(unsigned int);
#endif
	ktime			= 0;
//	CUT_SAFE_CALL(cutResetTimer(timer));
//	CUT_SAFE_CALL(cutStartTimer(timer));
	for (int i=0; i<NT; i++) {
	vlc_encode_kernel_sm64huff<<<grid_size, block_size, sm_size>>>(d_sourceData, d_codewords, d_codewordlens,  
																					d_cw32, d_cw32len, d_cw32idx, 
																d_destData, d_cindex); //testedOK2
	}
	cudaThreadSynchronize();
	CUT_CHECK_ERROR("Kernel execution failed\n");
//	CUT_SAFE_CALL(cutStopTimer(timer));
//	ktime += cutGetTimerValue(timer);
//	printf("GPU Encoding time (SM64HUFF): %f (ms)\n", ktime/NT);
	//////////////////* END KERNEL *///////////////////////////////////
#endif




#if 1
	unsigned int num_scan_elements = grid_size.x;
	preallocBlockSums(num_scan_elements);
	cudaMemset(d_destDataPacked, 0, mem_size);
	printf("Num_blocks to be passed to scan is %d.\n", num_scan_elements);
	prescanArray(d_cindex2, d_cindex, num_scan_elements);

	pack2<<< num_scan_elements/16, 16>>>((unsigned int*)d_destData, d_cindex, d_cindex2, (unsigned int*)d_destDataPacked, num_elements/num_scan_elements);
	CUT_CHECK_ERROR("Pack2 Kernel execution failed\n");
	deallocBlockSums();

	CUDA_SAFE_CALL(cudaMemcpy(destData, d_destDataPacked, mem_size, cudaMemcpyDeviceToHost));
	compare_vectors((unsigned int*)crefData, (unsigned int*)destData, num_ints);
	printdbg_data_bin("cpuout.txt", crefData, num_ints); 
	printdbg_data_bin("gpuout_dpt2.txt", destData, num_ints); 
	//CUDA_SAFE_CALL(cudaMemcpy(cindex2, d_cindex2, num_blocks*sizeof(int), cudaMemcpyDeviceToHost));
	//printdbg_data_int("blockscan.txt", cindex2, num_blocks); 
	//CUDA_SAFE_CALL(cudaMemcpy(cw32, d_cw32, mem_size, cudaMemcpyDeviceToHost));
	//CUDA_SAFE_CALL(cudaMemcpy(cw32len, d_cw32len, mem_size, cudaMemcpyDeviceToHost));
	//CUDA_SAFE_CALL(cudaMemcpy(cw32idx, d_cw32idx, mem_size, cudaMemcpyDeviceToHost));
	//printdbg_gpu_data_detailed2("gpuout_dpt2detailed.txt", cw32, cw32len, cw32idx, num_ints);
#endif

	free(sourceData); free(destData);  	free(codewords);  	free(codewordlens); free(cw32);  free(cw32len); free(crefData); 
	CUDA_SAFE_CALL(cudaFree(d_sourceData)); 	CUDA_SAFE_CALL(cudaFree(d_destData)); CUDA_SAFE_CALL(cudaFree(d_destDataPacked));
	CUDA_SAFE_CALL(cudaFree(d_codewords)); 		CUDA_SAFE_CALL(cudaFree(d_codewordlens));
	CUDA_SAFE_CALL(cudaFree(d_cw32)); 		CUDA_SAFE_CALL(cudaFree(d_cw32len)); 	CUDA_SAFE_CALL(cudaFree(d_cw32idx)); 
	CUDA_SAFE_CALL(cudaFree(d_cindex)); CUDA_SAFE_CALL(cudaFree(d_cindex2));
	free(cindex2);	
}

