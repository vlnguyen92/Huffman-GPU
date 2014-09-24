/*
 * Copyright Ana Balevic, Institute for Parallel and Distributed Systems (IPVS), University of Stuttgart. 2008.
 * All rights reserved.
 */

#ifndef _PABIO_KERNEL_H_
#define _PABIO_KERNEL_H_

//#include "cuda_constants.h"

/* PARALLEL PUT BITS IMPLEMENTATION (CUDA1.1+ compatible)
*  Set numbits in the destination word out[kc] starting from the position startbit
*  Implementation comments:
*  Second atomic operation actually sets these bits to the value stored in the codeword; the other bits are left unotuched
*  First atomic operation is a necessary prepration - we change only the bits that will be affected by the codeword to be written to 1s
*  in order for set bits to work with using atomicand.  
*  TODOs: benchmark performance 1) gm atomics vs sm atomics; 2) memset at init time vs. atomicOr
*/

__device__ void static put_bits_atomic0(unsigned int* out, unsigned int kc,
								unsigned int startbit, unsigned int numbits,
								unsigned int codeword) {
	unsigned int cw32 = codeword;
	unsigned int mask;
	unsigned int mask2;
	unsigned int restbits = 32-startbit-numbits;

	/* 1. Prepare the memory location */
	mask = 0xFFFFFFFF;  // -> 0000...00111100
	mask>>=(32-numbits); //fill in zeros at the front positions and sets the numbits bits to 1s
	mask<<=restbits;  //fill in zeros at the back positions
#ifndef MEMSET1 //Can remove this part if the contents of the memory are already set to all FFs
	atomicOr(&out[kc], mask);		//set 1s in the destination from startbit in the len of numbits
#endif

	/* 2. Write the codeword */
	mask2 = 0xFFFFFFFF;	// -> 1111...11abcd11
	mask2<<=numbits;     //set 1s in the front positions, and 0s at the last numbits in the codeword positions
	cw32|=mask2;         //in front are set 1s, in the back is set the codeword
	cw32 = (cw32<<restbits) | ~mask;	//shift to the appropriate position and set 1s at the end (|1111...11000011)
	atomicAnd(&out[kc], cw32);
}
//
///* TESTED OK */
//__global__ void put_bits_atomic_test_kernel(unsigned int* data, unsigned int* out) {
//	unsigned int kn = blockIdx.x*blockDim.x + threadIdx.x;
//	unsigned int val = data[kn];
//	unsigned int newval = val+1;
//	//put_bits_atomic(out, kn, 0, 32, newval);
//	put_bits_atomic(out, kn, 8, 8, newval);
//}
///* TESTED OK */
//__global__ void put_bits_atomic_sm_test_kernel(unsigned int* data, unsigned int* out) {
//	unsigned int kn = blockIdx.x*blockDim.x + threadIdx.x;
//	unsigned int k = threadIdx.x;
//	__shared__ unsigned int as[DATABLOCK_SIZE]; 
//	as[k] = 0x00000000;
//
//	unsigned int newval = data[kn]+1;
//
//	put_bits_atomic(as, k, 8, 8, newval);
//
//    __syncthreads();
//
//	out[kn] = as[k];
//
//}

#endif //ifndef _PABIO_KERNEL_H_