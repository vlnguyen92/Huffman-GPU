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
#include "print_helpers.h"
#include "comparison_helpers.h"
#include "stats_logger.h"
#include "load_data.h"

void runVLCTest(char *file_name, uint num_block_threads, uint num_blocks=1);

extern "C" void cpu_vlc_encode(unsigned int* indata, unsigned int num_elements, unsigned int* outdata, unsigned int *outsize, unsigned int *codewords, unsigned int* codewordlens);
extern "C" void cpu_vlc_encode_lame(unsigned int* sourceData, unsigned int num_elements, 
					unsigned int* destData, unsigned int *outsize, 
					unsigned int *codewords, unsigned int* codewordlens); 

int main(int argc, char* argv[]){
	//if(!InitCUDA()) { return 0;	}
	unsigned int num_block_threads = 256;
	if (argc > 1)
		for (int i=1; i<argc; i++)
			runVLCTest(argv[i], num_block_threads);
	else {runVLCTest(NULL, num_block_threads, 1024);	}
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
	////////LOAD DATA ///////////////
	uint	*sourceData =	(uint*) malloc(mem_size);
	uint	*destData   =	(uint*) malloc(mem_size);
	uint	*crefData   =	(uint*) malloc(mem_size);

	uint	*codewords	   = (uint*) malloc(NUM_SYMBOLS*symbol_type_size);
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
	printf("Codewords 32bit:\n");
//	print_array_in_hex(codewords, 256);
//	print_array<uint>(codewordlens, 256);
	//////// LOAD DATA ///////////////

	unsigned int sm_size; 


	unsigned int NT = 10; //number of runs for each execution time
	float ktime = 0.0f;
	unsigned int timer = 0;

	free(sourceData); free(destData);  	free(codewords);  	free(codewordlens); free(cw32);  free(cw32len); free(crefData); 
	free(cindex2);	
}



