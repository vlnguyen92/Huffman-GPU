/*
 * Copyright Ana Balevic, Institute for Parallel and Distributed Systems (IPVS), University of Stuttgart. 2008.
 * All rights reserved.
 */

#ifndef _VLC_SM1_KERNEL_H_
#define _VLC_SM1_KERNEL_H_

#include "parameters.h"
#include "pabio_kernels_v2.cu"
#ifdef SMATOMICS


/* 
*  PUT BITS IS PERFORMED ON THE SHARED MEMORY LOCATIONS
*  CONTENTS OF THE SHARED MEMORY ARE COPIED IN PARALLEL TO THE GLOBAL MEMORY
*  POSSIBLE OPTIMIZATIONS: - * reduce bank conflicts improve loads: use int data type for codewords and codewordlens, 
                           - * reduce sm usage: by using only 1 int array: first for cw_lens, then for prefix sums, and finally for the output data

*/

/* Variable Length Coding with Table Look-up
   1. CODEWORD LOOKUP DIRECTLY FROM THE GLOBAL MEMORY; DOES NOT REQUIRE SYNC THREADS BARRIER
   2. PARALLEL REDUCTION, REQUIRES 1 SYNC BEFORE AND 1 AFTER
   3. SM ATOMIC PABIO

    SM usage: 2x size of the input data
*/


__global__ static void vlc_encode_kernel_sm1(unsigned int* data,
								  const unsigned int* codewords, const unsigned int* codewordlens,
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

	extern __shared__ unsigned int as[];

	unsigned int *cw_lens = (unsigned int *)as;
	unsigned int *encdata = (unsigned int *)(as+blockDim.x);
	
	__shared__ unsigned int kcmax;

	/* Load the original data; Look-up the codeword in a codeword table */
	unsigned int val32	= data[kn];
	byte[0]	= (unsigned char)(val32>>24);
	byte[1] = (unsigned char)(val32>>16);
	byte[2] = (unsigned char)(val32>>8);
	byte[3] = (unsigned char) val32;

	for(unsigned int i=0; i<4;i++) {
		cw[i]		= codewords[byte[i]];
		cwlen[i]	= codewordlens[byte[i]];
		//codeword = (codeword<<8) | cw[i];  ///* Combine codewords for bytes into one codeword of max 32 bits */
		codeword = (codeword<<cwlen[i]) | cw[i];
		codewordlen+=cwlen[i];
	}

	cw_lens[k] = codewordlen;
	encdata[k] = 0U;

	__syncthreads();


	/* Prefix sum of codeword lengths (denoted in bits) [inplace implementation] */ 
	unsigned int offset = 1;

    /* Build the sum in place up the tree */
    for (unsigned int d = (blockDim.x)>>1; d > 0; d >>= 1)  {
        __syncthreads();
        if (k < d)   {
            unsigned char ai = offset*(2*k+1)-1;
            unsigned char bi = offset*(2*k+2)-1;
            cw_lens[bi] += cw_lens[ai];
        }
        offset *= 2;
    }

    /* scan back down the tree */
    /* clear the last element */
    if (k == 0) cw_lens[blockDim.x - 1] = 0;    

    // traverse down the tree building the scan in place
    for (unsigned int d = 1; d < blockDim.x; d *= 2)    {
        offset >>= 1;
        __syncthreads();
        if (k < d)   {
            unsigned char ai = offset*(2*k+1)-1;
            unsigned char bi = offset*(2*k+2)-1;
            unsigned int t   = cw_lens[ai];
            cw_lens[ai]  = cw_lens[bi];
            cw_lens[bi] += t;
        }
    }
    __syncthreads();

	if (k == blockDim.x-1)
		outidx[blockIdx.x] = cw_lens[k] + codewordlen;


#ifdef TESTING
	//TODO: remove (for testing only)
	cw32[kn]	= codeword; 
	cw32len[kn] = codewordlen;
	cw32idx[kn] = cw_lens[k];
#endif
	////////////////////////////////////////////////////////////////////////

	/* Calculate kc_index-es as (prefix sum value) ps_cw_lens/32 if the output array is composed of 32-bit words */
	kc = cw_lens[k]>>5;  //kc = ps_cw_lens[k]>>5; // div 32   

	/* Combine codewords matching the symbols for the write to the location kc in the output array */

	/* Detect if the codeword would overflow the word boundary */
	unsigned int startbit = cw_lens[k]%32; //cw_lens[k]&31; //ps_cw_lens[4*k+i]%32; //TODO: optimization - replace with a % n == a & (n-1) if n is a power of 2
	unsigned int numbits  = codewordlen;
	unsigned int overflow = (startbit + numbits > 32) ? 1:0;  /* 0 = Good situation: the complete codewords fits onto the destination location kc , 1= overflows to the next destination index kc+1*/
	numbits =  overflow? (32-startbit) : numbits;   //if there's overflow, just take the first part of the codeword which fits in, numbits from startbit to 32-bit boundary

	if (threadIdx.x == blockDim.x-1) kcmax = kc + overflow; //find out the number of coded words, and broadcast kcmax value

	unsigned int codeword_part = codeword>>(codewordlen - numbits);
	put_bits_atomic2(encdata, kc, startbit, numbits, codeword_part);


	if (overflow) {  
		numbits = codewordlen - numbits;
		codeword_part = codeword & ((1<<numbits)-1);
		put_bits_atomic2(encdata, kc+1, 0, numbits, codeword_part);
	} 

	__syncthreads();
	/* Write to the global memory */
#if 1
	if (k<=kcmax) out[kn] = encdata[k];
#else
	out[kn] = (k<=kcmax)? encdata[k]:0;
#endif
}
//////////////////////////////////////////////////////////////////////////////								  
#endif


#endif