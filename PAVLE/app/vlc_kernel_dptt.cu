/*
 * PAVLE - Parallel Variable-Length Encoder for CUDA
 *
 * Copyright (C) 2009 Ana Balevic <ana.balevic@gmail.com> and Tjark Bringewat <golvellius@gmx.net>
 * All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify it under the terms of the
 * MIT License. Read the full licence: http://www.opensource.org/licenses/mit-license.php
 *
 * If you find this program useful, please contact me and reference PAVLE home page in your work.
 * 
 */

#ifndef _VLC_DPTT_H_
#define _VLC_DPTT_H_

#include "parameters.h"
#include "pabio_kernels_v2.cu"

__global__ static void vlc_encode_kernel_dptt(unsigned int *data, unsigned int *codewords, unsigned int *codewordlens,
									#ifdef TESTING
										unsigned int *data32, unsigned int *cw32len, unsigned int *cw32idx,
									#endif
										unsigned int *out, unsigned int *outidx){

	unsigned int kn = (blockIdx.x*blockDim.x + threadIdx.x) * DPT;
	unsigned int k = threadIdx.x;
	unsigned int kc;

	unsigned int cw[4] = {0,0,0,0};
	unsigned int cwlen = 0;
	unsigned int cwlen2 = 0;
	unsigned int val32, tmpcw, tmpcwlen;
	unsigned char byte;

	extern __shared__ unsigned int as[];
	unsigned int *cwlens = as + NUM_SYMBOLS;
	unsigned int *cw_lens = as + 2*NUM_SYMBOLS;

	// Handle different numbers of threads (256 threads per block is ideal)
	if (k < NUM_SYMBOLS) {
		as[k] = codewords[k];
		cwlens[k] = codewordlens[k];
		for (unsigned int i=1; i<NUM_SYMBOLS/blockDim.x; i++) {
			as[k+i*blockDim.x] = codewords[k+i*blockDim.x];
			cwlens[k+i*blockDim.x] = codewordlens[k+i*blockDim.x];
		}
	}
	__syncthreads();

	for(unsigned int i=0; i<DPT; i++) {
		val32 = data[kn+i];								// load original data
		for(unsigned int b=0; b<4; b++) {
			byte =(unsigned char) (val32>>(3-b)*8);
			tmpcw	 = as[byte];						// code word for current byte
			tmpcwlen = cwlens[byte];					// code-word length for current byte
			cwlen  += tmpcwlen;							// overall code-word length so far
			cwlen2 += tmpcwlen;							// code-word length within the current dword so far
			if (cwlen2 > 31) {
				cwlen2 -= 32;
				cw[cwlen/32-1] |= tmpcw >> cwlen2;
			}
			cw[cwlen/32] |= tmpcw << 32-cwlen2;
		}
	}
	cw[cwlen/32] >>= 32-cwlen2;
	cw_lens[k] = cwlen;
	__syncthreads();

	// Prefix sum of codeword lengths (denoted in bits) [inplace implementation]
	unsigned int offset = 1;

    // Build the sum in place up the tree
	for (unsigned int d=blockDim.x>>1; d>0; d>>=1) {
		if (k < d) {
			unsigned int ai = offset*(2*k+1)-1;
			unsigned int bi = offset*(2*k+2)-1;
			cw_lens[bi] += cw_lens[ai];
		}
		offset *= 2;
		__syncthreads();
	}

	// scan back down the tree
	// clear the last element
	if (k == 0) cw_lens[blockDim.x-1] = 0;
	__syncthreads();

	// traverse down the tree building the scan in place
	for (unsigned int d=1; d<blockDim.x; d*=2) {
		offset >>= 1;
		if (k < d) {
			unsigned int ai = offset*(2*k+1)-1;
			unsigned int bi = offset*(2*k+2)-1;
			unsigned int t  = cw_lens[ai];
			cw_lens[ai]  = cw_lens[bi];
			cw_lens[bi] += t;
		}
		__syncthreads();
	}
	/////////////////////////
	if (k == blockDim.x-1)
		outidx[blockIdx.x] = cw_lens[k] + cwlen;

	#ifdef TESTING
	data32[kn] = val32;
	cw32len[kn] = cwlen;
	cw32idx[kn] = cw_lens[k];
	#endif

	// Write the codes
	unsigned int startbit, numbits, overflow, codeword_part, n;
	#ifndef SMATOMICS
		#define OUTPUT out
		kc = blockIdx.x*blockDim.x*DPT + cw_lens[k]/32;
	#else
		#define OUTPUT encdata
		kc = cw_lens[k]/32;
		unsigned int *encdata = cw_lens + blockDim.x;
		for (unsigned int i=0; i<DPT; i++)
			encdata[k*DPT+i] = 0;
		__syncthreads();
	#endif
	startbit = cw_lens[k] % 32;

	n = 0;
	while (cwlen > 0) {
		cwlen2		= cwlen > 32 ? 32 : cwlen;
		overflow	= startbit+cwlen2 > 32 ? 1 : 0;
		numbits		= overflow ? 32-startbit : cwlen2;
		codeword_part = cw[n] >> cwlen2-numbits;
		atomicOr(&OUTPUT[kc], codeword_part << 32-numbits-startbit);
		kc++;
		if (overflow) {
			numbits = cwlen2 - numbits;
			codeword_part = cw[n] << 32-numbits;
			atomicOr(&OUTPUT[kc], codeword_part);
			startbit = numbits;
		}
		cwlen -= cwlen2;
		n++;
	}
	#ifdef SMATOMICS
		__syncthreads();

		//for (unsigned int i=0; i<DPT; i++)
		//	out[kn+i] = encdata[k*DPT+i];


		//Copy only the encoded data from cwbuff to the global memory!
		unsigned int lastidx = outidx[blockIdx.x]/32;

		for (unsigned int r=0; r<DPT; r++) 
			if ((blockDim.x*r+k)<= lastidx) out[blockIdx.x*blockDim.x*DPT+blockDim.x*r+k] = encdata[blockDim.x*r+k]; 


	#endif
}

#endif
