/*
 * Copyright Ana Balevic, Institute for Parallel and Distributed Systems (IPVS), University of Stuttgart. 2008.
 * All rights reserved.
 */

#ifndef _VLC_SM2HUFF_KERNEL_H_
#define _VLC_SM2HUFF_KERNEL_H_

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


__global__ static void vlc_encode_kernel_sm2_huff(unsigned int* data,
								  const unsigned int* gm_codewords, const unsigned int* gm_codewordlens,
							#ifdef TESTING
								  unsigned int* cw32, unsigned int* cw32len, unsigned int* cw32idx, 
							#endif
								  unsigned int* out, unsigned int *outidx){

	unsigned int kn = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int k = threadIdx.x;

	unsigned long long cw64 =0;
	unsigned int val32, codewordlen = 0;
	unsigned char tmpbyte, tmpcwlen;
	unsigned int tmpcw;

	extern __shared__ unsigned int sm[];
	__shared__ unsigned int kcmax;

	unsigned int* codewords		= (unsigned int*) sm; 
	unsigned int* codewordlens	= (unsigned int*)(sm+NUM_SYMBOLS); 
	unsigned int* as			= (unsigned int*)(sm+2*NUM_SYMBOLS); 

	/* Load the codewords and the original data*/
	codewords[k]	= gm_codewords[k];
	codewordlens[k] = gm_codewordlens[k];
	val32			= data[kn];
	__syncthreads();

	for(unsigned int i=0; i<4;i++) {
		tmpbyte = (unsigned char)(val32>>((3-i)*8));
		tmpcw = codewords[tmpbyte];
		tmpcwlen = codewordlens[tmpbyte];
		cw64 = (cw64<<tmpcwlen) | tmpcw;
		codewordlen+=tmpcwlen;
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
	unsigned int kc, startbit, write_bits, mask32, n=0;
	kc = as[k]>>5;  
	startbit = as[k]&31; 
	as[k] =  0U;
	__syncthreads();

	while (codewordlen > 0)  {
		write_bits = min(32-startbit, codewordlen);
		mask32 = (unsigned int)(cw64>>(codewordlen-write_bits)); 	//get first bits_available=write_bits bits from the cw64
		mask32<<=(32-startbit-write_bits);	//position them (if needed)
		if (write_bits==32)	
			as[kc+n] = mask32; 
		else 
			atomicOr(&as[kc+n], mask32);	 	//output the bits:	
		cw64 = cw64 & ((1<<(codewordlen-write_bits))-1);		//erase them	
		codewordlen-=write_bits;
		startbit = 0;	n++; 		//move pointers
	}

	if (threadIdx.x == blockDim.x-1) kcmax = kc + n;

	__syncthreads();

	if (k<=kcmax) out[kn] = as[k];

}
//////////////////////////////////////////////////////////////////////////////								  
#endif

#endif