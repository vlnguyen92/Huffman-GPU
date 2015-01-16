/*
 * PAVLE - Parallel Variable-Length Encoder for CUDA
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


#ifndef _VLC_SM32_KERNEL_H_
#define _VLC_SM32_KERNEL_H_

#include "parameters.h"
#include "pabio_kernels_v2.cu"

#ifdef SMATOMICS

/* PAVLE CHARACTERISTICS:
   1. CACHE CW_LUT INTO SM, LOAD AS 2 INT ARRAYS
   2. PARALLEL PREFIX SUM
   3. PARALLEL BIT I/O USING SHARED-MEMORY ATOMIC OPERATIONS (COMAPTIBLE WITH CUDA1.3+)
   
   ASSUMPTIONS:
   -	COMBINED CODEWORDS MUST NOT BE LONGER THEN THE ORIGINAL DATA (4 CODEWORDS <= 32bits)
   -	NUMBER OF THREADS PER BLOCK IS 256; IF YOU WANT TO PLAY WITH DIFFERENT NUMBERS, THE CW CACHING SHOULD BE MODIFIED (SEE DPT* KERNELS FOR SOLUTION) 

*/

__global__ static void vlc_encode_kernel_sm32(unsigned int* data,
								  const unsigned int* gm_codewords, const unsigned int* gm_codewordlens,
							#ifdef TESTING
								  unsigned int* cw32, unsigned int* cw32len, unsigned int* cw32idx, 
							#endif
								  unsigned int* out, unsigned int *outidx){

	unsigned int kn = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int k = threadIdx.x;

	unsigned char byte[4];
	unsigned int  cw[4], cwlen[4];

	unsigned int codeword = 0x00000000;
	unsigned int codewordlen = 0;

	extern __shared__ unsigned int sm[];
	__shared__ unsigned int kcmax;


#ifdef CACHECWLUT
	unsigned int* codewords = (unsigned int*) sm; 
	unsigned int* codewordlens = (unsigned int*)(sm+NUM_SYMBOLS); 
	unsigned int* as = (unsigned int*)(sm+2*NUM_SYMBOLS); 

	unsigned int val32	= data[kn];
	byte[0]	= (unsigned char)(val32>>24);
	byte[1] = (unsigned char)(val32>>16);
	byte[2] = (unsigned char)(val32>>8);
	byte[3] = (unsigned char) val32;

	codewords[k]	 = gm_codewords[k];
	codewordlens[k]  = gm_codewordlens[k];
	__syncthreads();

	#pragma unroll 4
	for(unsigned int i=0; i<4;i++) {
		cw[i]		= codewords[byte[i]];
		cwlen[i]	= codewordlens[byte[i]];
		codeword = (codeword<<cwlen[i]) | cw[i];
		codewordlen+=cwlen[i];
	}
#else
	unsigned int* as	= (unsigned int*) sm;

	unsigned int val32	= data[kn];
	byte[0]	= (unsigned char)(val32>>24);
	byte[1] = (unsigned char)(val32>>16);
	byte[2] = (unsigned char)(val32>>8);
	byte[3] = (unsigned char) val32;

	#pragma unroll 4
	for(unsigned int i=0; i<4;i++) {
		cw[i] = gm_codewords[byte[i]];
		cwlen[i] = gm_codewordlens[byte[i]];
		codeword = (codeword<<cwlen[i]) | cw[i];
		codewordlen+=cwlen[i];
	}
#endif

	as[k] = codewordlen;
	__syncthreads();


	/* Prefix sum of codeword lengths (denoted in bits) [inplace implementation] */ 
	unsigned int offset = 1;

    /* Build the sum in place up the tree */
    for (unsigned int d = (blockDim.x)>>1; d > 0; d >>= 1)  {
        __syncthreads();
        if (k < d)   {
            unsigned char ai = offset*(2*k+1)-1;
            unsigned char bi = offset*(2*k+2)-1;
            as[bi] += as[ai];
        }
        offset *= 2;
    }

    /* scan back down the tree */
    /* clear the last element */
    if (k == 0) as[blockDim.x - 1] = 0;    

    // traverse down the tree building the scan in place
    for (unsigned int d = 1; d < blockDim.x; d *= 2)    {
        offset >>= 1;
        __syncthreads();
        if (k < d)   {
            unsigned char ai = offset*(2*k+1)-1;
            unsigned char bi = offset*(2*k+2)-1;
            unsigned int t   = as[ai];
            as[ai]  = as[bi];
            as[bi] += t;
        }
    }
	__syncthreads();

	if (k == blockDim.x-1) {
		outidx[blockIdx.x] = as[k] + codewordlen;
		kcmax = (as[k] + codewordlen)/32;
	}

	/* Write the codes */
	unsigned int kc = as[k]>>5;  
	unsigned int startbit = as[k]&31; 
	unsigned int numbits  = codewordlen;
	unsigned int overflow = (startbit + numbits > 32) ? 1:0;  
	numbits =  overflow? (32-startbit) : numbits;   

	as[k] =  0U;
	__syncthreads();

	/* Combine codewords matching the symbols for the write to the location kc in the output array */
	unsigned int codeword_part = codeword>>(codewordlen - numbits);
	put_bits_atomic2(as, kc, startbit, numbits, codeword_part);

	if (overflow) {  
		numbits = codewordlen - numbits;
		codeword_part = codeword & ((1<<numbits)-1);
		put_bits_atomic2(as, kc+1, 0, numbits, codeword_part);
	} 

	__syncthreads();
	if (k<=kcmax) out[kn] = as[k];

}
							  
#endif
#endif