/*
 * Copyright Ana Balevic, Institute for Parallel and Distributed Systems (IPVS), University of Stuttgart. 2008.
 * All rights reserved.
 */

#ifndef _VLC_GM0_HUFF_KERNEL_H_
#define _VLC_GM0_HUFF_KERNEL_H_


#include "parameters.h"
#include "pabio_kernels_v0.cu"
#include "pabio_kernels_v2.cu"

/* Variable Length Coding with Table Look-up
   1. CODEWORD LOOKUP DIRECTLY FROM THE GLOBAL MEMORY; DOES NOT REQUIRE SYNC THREADS BARRIER
   2. PARALLEL REDUCTION, REQUIRES 1 SYNC BEFORE AND 1 AFTER
   3. GM ATOMIC PABIO
*/


/* Variable Length Coding with Table Look-up */
__global__ static void vlc_encode_kernel_gm0_huff(unsigned int* data,
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

	extern __shared__ unsigned int cw_lens[]; 

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
		//codewordlen +=8;
		codeword = (codeword<<cwlen[i]) | cw[i];
		codewordlen+=cwlen[i];
	}

	cw_lens[k] = codewordlen;
	__syncthreads();


	/* Prefix sum of codeword lengths (denoted in bits) [inplace implementation] */ 
	unsigned int offset = 1;

    /* Build the sum in place up the tree */
    for (unsigned int d = (blockDim.x)>>1; d > 0; d >>= 1)  {
        __syncthreads();
        if (k < d)   {
            unsigned int ai = offset*(2*k+1)-1;
            unsigned int bi = offset*(2*k+2)-1;
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
            unsigned int ai = offset*(2*k+1)-1;
            unsigned int bi = offset*(2*k+2)-1;
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
	cw32[kn] = codeword; 
	cw32len[kn] = codewordlen;
	cw32idx[kn] = (unsigned int)cw_lens[k];
	#endif

	/* Write the codes */
	kc = blockIdx.x*blockDim.x +(cw_lens[k]/32); //kc = blockIdx.x*blockDim.x + cw_lens[k]>>5;  <--DOES NOT WORK!
	unsigned int startbit = cw_lens[k]%32; 
	unsigned int numbits  = codewordlen;
	unsigned int overflow = (startbit + numbits > 32) ? 1:0;  
	numbits =  overflow? (32-startbit) : numbits;   

	/* Combine codewords matching the symbols for the write to the location kc in the output array */
	unsigned int codeword_part = codeword>>(codewordlen - numbits);
	//put_bits_atomic0(out, kc, startbit, numbits, codeword_part);
	put_bits_atomic2(out, kc, startbit, numbits, codeword_part);

	/* If the codeword overflows to the next destination index, take care of the remaining bits */
	if (overflow) {  /* Write a second part of the codeword to the location kc+1 starting from the bit position 0 */
		numbits = codewordlen - numbits;
		//codeword_part = codeword<<(32-numbits); 
		//codeword_part >>=(32-numbits);
		codeword_part = (codeword) & ((1<<numbits)-1); 
		//put_bits_atomic0(out, kc+1, 0, numbits, codeword_part);
		put_bits_atomic2(out, kc+1, 0, numbits, codeword_part);
	} 

}
							  
#endif


