#ifndef _SCAN_KERNEL_H_
#define _SCAN_KERNEL_H_

__global__ static void scan_kernel(unsigned int* data, unsigned int* pscw32) {

	unsigned int kn = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int k = threadIdx.x;

	extern __shared__ unsigned int cw_lens[]; 

	cw_lens[k] = data[kn];
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

}
							  
#endif


