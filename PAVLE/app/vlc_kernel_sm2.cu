/*
 * Copyright Ana Balevic, Institute for Parallel and Distributed Systems (IPVS), University of Stuttgart. 2008.
 * All rights reserved.
 */

#ifndef _VLC_SM2_KERNEL_H_
#define _VLC_SM2_KERNEL_H_

#include "parameters.h"
#include "pabio_kernels_v2.cu"

#ifdef SMATOMICS

/* Variable Length Coding with Table Look-up
   1. CACHE CW_LUT INTO SM, LOAD AS 2 INT ARRAYS; EACH THREAD LOADS 1 CW
   2. PARALLEL REDUCTION, REQUIRES 1 SYNC BEFORE AND 1 AFTER
   3. SM ATOMIC PABIO

   SM usage: 1x size of the input data (REUSE) + size of CWLUT
   corresponds to vlc_encode_kernel_gm2:
*/


__global__ static void vlc_encode_kernel_sm2(unsigned int* data,
								  const unsigned int* gm_codewords, const unsigned int* gm_codewordlens,
							#ifdef TESTING
								  unsigned int* cw32, unsigned int* cw32len, unsigned int* cw32idx, 
							#endif
								  unsigned int* out, unsigned int *outidx){

	unsigned int kn = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int k = threadIdx.x;
	unsigned int kc;

	unsigned char byte[4];
	unsigned int  cw[4], cwlen[4];

	unsigned int codeword = 0x00000000;
	unsigned int codewordlen = 0;

	// extern __shared__ unsigned int cw_lens[]; 
	// extern __shared__ unsigned int encdata[]; 
	//unsigned int* codewords = (unsigned int*) as; 
	//unsigned int* codewordlens = (unsigned int*)(as+NUM_SYMBOLS); 
	//unsigned int* cw_lens = (unsigned int*)(as+2*NUM_SYMBOLS); 

	extern __shared__ unsigned int sm[];
	__shared__ unsigned int kcmax;
	unsigned int* codewords = (unsigned int*) sm; 
	unsigned int* codewordlens = (unsigned int*)(sm+NUM_SYMBOLS); 
	unsigned int* as = (unsigned int*)(sm+2*NUM_SYMBOLS); 

	codewords[k]	 = gm_codewords[k];
	codewordlens[k]  = gm_codewordlens[k];
	/* Load the original data; Look-up the codeword in a codeword table */
	unsigned int val32	= data[kn];
	byte[0]	= (unsigned char)(val32>>24);
	byte[1] = (unsigned char)(val32>>16);
	byte[2] = (unsigned char)(val32>>8);
	byte[3] = (unsigned char) val32;
	__syncthreads();

	#pragma unroll 4
	for(unsigned int i=0; i<4;i++) {
		cw[i]		= codewords[byte[i]];
		cwlen[i]	= codewordlens[byte[i]];
		//codeword = (codeword<<8) | cw[i];  ///* Combine codewords for bytes into one codeword of max 32 bits */
		codeword = (codeword<<cwlen[i]) | cw[i];

		codewordlen+=cwlen[i];
	}

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

	if (k == blockDim.x-1) 
		outidx[blockIdx.x] = as[k] + codewordlen;
	
	/* Write the codes */

	kc = as[k]>>5;  
	unsigned int startbit = as[k]&31; 
	unsigned int numbits  = codewordlen;
	unsigned int overflow = (startbit + numbits > 32) ? 1:0;  
	numbits =  overflow? (32-startbit) : numbits;   

	if (threadIdx.x == blockDim.x-1) kcmax = kc + overflow; 

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
//////////////////////////////////////////////////////////////////////////////								  
#endif

#endif
