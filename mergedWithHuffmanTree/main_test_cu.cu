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
#include "cuda_helpers.h"
#include "print_helpers.h"
#include "comparison_helpers.h"
#include "stats_logger.h"
#include "load_data.h"
#include "parameters.h"
#include <iostream>
//#include "vlc_kernel_gm32.cu"
//#include "vlc_kernel_sm32.cu"
#include "vlc_kernel_sm64huff.cu"
//#include "vlc_kernel_dpt.cu"
//#include "vlc_kernel_dptt.cu"
//#include "scan_kernel.cu"
#include "scan.cu"
#include "pack_kernels.cu"
#include "cpuencode.cpp"

using namespace std;

void runVLCTest(char *file_name, uint num_block_threads, uint num_blocks=1);

extern "C" void cpu_vlc_encode(unsigned int* indata, unsigned int num_elements, unsigned int* outdata, unsigned int *outsize, unsigned int *codewords, unsigned int* codewordlens);
extern "C" void cpu_vlc_encode_lame(unsigned int* sourceData, unsigned int num_elements, 
        unsigned int* destData, unsigned int *outsize, 
        unsigned int *codewords, unsigned int* codewordlens); 

int main(int argc, char* argv[]){
    if(!InitCUDA()) { return 0;	}
    unsigned int num_block_threads = 256;
    if (argc > 1)
        for (int i=1; i<argc; i++)
            runVLCTest(argv[i], num_block_threads);
    else {	runVLCTest(NULL, num_block_threads, 1024);	}
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
    ////////LOAD DATA ///////////////
    //    uint	*sourceData =	(uint*) malloc(mem_size);
    uint *sourceData;
    cudaHostAlloc((void**)&sourceData,mem_size,cudaHostAllocDefault);
    uint	*destData   =	(uint*) malloc(mem_size);
    uint	*crefData   =	(uint*) malloc(mem_size);

    //    cout << "HERE" << endl;
    uint	*codewords	   = (uint*) malloc(NUM_SYMBOLS*symbol_type_size);
    uint	*codewordlens  = (uint*) malloc(NUM_SYMBOLS*symbol_type_size);

    //    uint	*cw32 =		(uint*) malloc(mem_size);
    //    uint	*cw32len =	(uint*) malloc(mem_size);
    //    uint	*cw32idx =	(uint*) malloc(mem_size);

    uint	*cindex2=	(uint*) malloc(num_blocks*sizeof(int));

    memset(sourceData,   0, mem_size);
    memset(destData,     0, mem_size);
    memset(crefData,     0, mem_size);
    //    memset(cw32,         0, mem_size);
    //    memset(cw32len,      0, mem_size);
    //    memset(cw32idx,      0, mem_size);
    memset(codewords,    0, NUM_SYMBOLS*symbol_type_size);
    memset(codewordlens, 0, NUM_SYMBOLS*symbol_type_size);
    memset(cindex2, 0, num_blocks*sizeof(int));
    //////// LOAD DATA ///////////////
    loadData(file_name, sourceData, codewords, codewordlens, num_elements, mem_size, H);
    //    for(int i = 0; i < 256; i++)
    //    {
    //        if(codewordlens[i]) 
    //        {
    //            cout << i << " " << codewords[i] << " "; 
    //            //            printBits(codewords[i],codewordlens[i]);
    //            cout << endl;
    //        }
    //    }
    //printf("Codewords 32bit:\n");
    //print_array_in_hex(codewords, 256);
    //print_array<uint>(codewordlens, 256);
    //////// LOAD DATA ///////////////

    unsigned int	*d_sourceData, *d_destData, *d_destDataPacked;
    unsigned int	*d_codewords, *d_codewordlens;
    unsigned int	 *d_cindex, *d_cindex2;
    //    unsigned int   *d_cw32len,*d_cw32,*d_cw32idx;

    // unsigned int sm_block_size_for_cwlens = num_block_threads * sizeof(unsigned int);
    unsigned int sm_size; 


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
    //cpu_vlc_encode_lame((unsigned int*)sourceData, num_elements, (unsigned int*)crefData,  &refbytesize, codewords, codewordlens);
    //	CUT_SAFE_CALL(cutStopTimer(timer));
    //	ktime = cutGetTimerValue(timer);
    //    printf("CPU Encoding time (CPU): %f (ms)\n", ktime);
    printf("CPU Encoded to %d [B]\n", refbytesize);
    unsigned int num_ints = refbytesize/4 + ((refbytesize%4 ==0)?0:1);
    //////////////////* END CPU *///////////////////////////////////


#if 0
    //////////////////* GM32 KERNEL *///////////////////////////////////
    grid_size.x		= num_blocks;
    block_size.x	= num_block_threads;
    sm_size			= block_size.x*sizeof(unsigned int);
#ifdef CACHECWLUT
    sm_size			= 2*NUM_SYMBOLS*sizeof(int) + block_size.x*sizeof(unsigned int);
#endif
    ktime			= 0;
    CUT_SAFE_CALL(cutResetTimer(timer));
    CUT_SAFE_CALL(cutStartTimer(timer));
    vlc_encode_kernel_gm32<<<grid_size, block_size, sm_size>>>(d_sourceData, d_codewords, d_codewordlens,  
#ifdef TESTING
            d_cw32, d_cw32len, d_cw32idx, 
#endif
            d_destData, d_cindex); //testedOK2
    cudaThreadSynchronize();
    CUT_CHECK_ERROR("Kernel execution failed\n");
    CUT_SAFE_CALL(cutStopTimer(timer));
    ktime += cutGetTimerValue(timer);
    printf("GPU Encoding time (GM32): %f (ms)\n", ktime);
    //////////////////* END KERNEL *///////////////////////////////////
#endif


#if 0
    //////////////////* SM32 KERNEL *///////////////////////////////////
    grid_size.x		= num_blocks;
    block_size.x	= num_block_threads;
    sm_size			= block_size.x*sizeof(unsigned int);
#ifdef CACHECWLUT
    sm_size			= 2*NUM_SYMBOLS*sizeof(int) + block_size.x*sizeof(unsigned int);
#endif
    ktime			= 0;
    CUT_SAFE_CALL(cutResetTimer(timer));
    CUT_SAFE_CALL(cutStartTimer(timer));
    for (int i=0; i<NT; i++) {
        vlc_encode_kernel_sm32<<<grid_size, block_size, sm_size>>>(d_sourceData, d_codewords, d_codewordlens,  
#ifdef TESTING
                d_cw32, d_cw32len, d_cw32idx, 
#endif
                d_destData, d_cindex); //testedOK2
    }
    cudaThreadSynchronize();
    CUT_CHECK_ERROR("Kernel execution failed\n");
    CUT_SAFE_CALL(cutStopTimer(timer));
    ktime += cutGetTimerValue(timer);
    printf("GPU Encoding time (SM32): %f (ms)\n", ktime/NT);
    //////////////////* END KERNEL *///////////////////////////////////
#endif
    unsigned int partSize = mem_size/32;
    unsigned int numIter = mem_size/partSize;//clamp this later
    unsigned int offset = partSize/symbol_type_size;
    cudaEvent_t     start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    CUDA_SAFE_CALL(cudaMalloc((void**) &d_sourceData,		  partSize));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_destData,			  partSize));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_destDataPacked,	  partSize));

    CUDA_SAFE_CALL(cudaMalloc((void**) &d_codewords,		  NUM_SYMBOLS*symbol_type_size));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_codewordlens,		  NUM_SYMBOLS*symbol_type_size));

    //    CUDA_SAFE_CALL(cudaMalloc((void**) &d_cw32,				  partSize));
    //    CUDA_SAFE_CALL(cudaMalloc((void**) &d_cw32len,			  partSize));
    //    CUDA_SAFE_CALL(cudaMalloc((void**) &d_cw32idx,			  partSize));

    unsigned int new_num_elements = partSize/symbol_type_size;
    unsigned int new_blocks = new_num_elements/num_block_threads;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_cindex,         new_blocks*sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_cindex2,        new_blocks*sizeof(unsigned int)));
    dim3 grid_size(new_blocks,1,1);
    dim3 block_size(num_block_threads, 1, 1);

    //   uint	*partialDestData   =	(uint*) malloc(partSize);
    //   memset(partialDestData,     0, partSize);

    CUDA_SAFE_CALL(cudaMemcpy(d_codewords,		codewords,		NUM_SYMBOLS*symbol_type_size,	cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_codewordlens,	codewordlens,	NUM_SYMBOLS*symbol_type_size,	cudaMemcpyHostToDevice));
    ( cudaEventRecord( start, 0 ) );

    for(int i = 0; i < num_elements; i+=new_num_elements)    
    {
        //        cout << i << endl;
        CUDA_SAFE_CALL(cudaMemcpy(d_sourceData,		sourceData+i,		partSize,		cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(d_destData,		destData+i,		partSize,		cudaMemcpyHostToDevice));
#if 1
        //////////////////* SM64HUFF KERNEL *///////////////////////////////////
        grid_size.x		= new_blocks;
        block_size.x	= num_block_threads;
        sm_size			= block_size.x*sizeof(unsigned int);
#ifdef CACHECWLUT
        sm_size			= 2*NUM_SYMBOLS*sizeof(int) + block_size.x*sizeof(unsigned int);
#endif
        ktime			= 0;
        //	CUT_SAFE_CALL(cutResetTimer(timer));
        //	CUT_SAFE_CALL(cutStartTimer(timer));
//        for (int i=0; i<NT; i++) {
            vlc_encode_kernel_sm64huff<<<grid_size, block_size, sm_size>>>(d_sourceData, d_codewords, d_codewordlens,  
                    //                d_cw32, d_cw32len, d_cw32idx, 
                    d_destData, d_cindex); //testedOK2
 //       }
        CUDA_SAFE_CALL(cudaThreadSynchronize());
        CUT_CHECK_ERROR("Kernel execution failed\n");
        //	CUT_SAFE_CALL(cutStopTimer(timer));
        //	ktime += cutGetTimerValue(timer);
        printf("GPU Encoding time (SM64HUFF): %f (ms)\n", ktime/NT);
        //////////////////* END KERNEL *///////////////////////////////////
#endif

        unsigned int num_scan_elements = grid_size.x;
        preallocBlockSums(num_scan_elements);
        cudaMemset(d_destDataPacked, 0, partSize);
        printf("Num_blocks to be passed to scan is %d.\n", num_scan_elements);
        prescanArray(d_cindex2, d_cindex, num_scan_elements);

        //this works for kernels GM1, GM2, SM1, sm2
        //pack2<<<num_blocks/ENCODE_THREADS, ENCODE_THREADS>>>((unsigned int*)d_destData, d_cindex, d_cindex2, (unsigned int*)d_destDataPacked);
        //pack2<<<grid_size/16, 16>>>((unsigned int*)d_destData, d_cindex, d_cindex2, (unsigned int*)d_destDataPacked);

        pack2<<< num_scan_elements/16, 16>>>((unsigned int*)d_destData, d_cindex, d_cindex2, (unsigned int*)d_destDataPacked, new_num_elements/num_scan_elements);
        CUT_CHECK_ERROR("Pack2 Kernel execution failed\n");
        deallocBlockSums();
        CUDA_SAFE_CALL(cudaMemcpy(destData+i, 
                    //partialDestData, 
                    d_destDataPacked, 
                    partSize, 
                    cudaMemcpyDeviceToHost));

        //       memcpy(destData + (i * offset),partialDestData,partSize);

//        destData += offset;
        //       memcpy(destData,partialDestData,partSize);
 //       sourceData += offset;
        //        compare_vectors((unsigned int*)partDestData,(unsigned int*)destData, partSize/4 + ((partSize%4 ==0)?0:1)); 
    }
    cudaEventRecord( stop, 0 ) ;
    cudaEventSynchronize( stop ) ;
    float   elapsedTime;
    cudaEventElapsedTime( &elapsedTime,
            start, stop ) ;
    printf( "Total time to encode %d Mb:  %3.1f ms\n", mem_size/1024,elapsedTime );


//    destData-=(numIter) * offset;
    //    compare_vectors((unsigned int*)crefData, (unsigned int*)destData, num_ints);
//    sourceData-=(numIter) * offset;
    printdbg_data_bin("cpuout.txt", crefData, num_ints); 
    printdbg_data_bin("gpuout_dpt2.txt", destData, num_elements); 

#if 0
    //////////////////* DPT KERNEL *///////////////////////////////////
    //grid_size.x		= num_blocks;
    //block_size.x	= num_block_threads/DPT;
    //sm_size			= 2*NUM_SYMBOLS*sizeof(int) + block_size.x*sizeof(unsigned int)+ block_size.x*DPT*sizeof(unsigned int);
    block_size.x	= num_block_threads;
    grid_size.x		= num_blocks/DPT;
    sm_size			= block_size.x*sizeof(unsigned int)+ block_size.x*DPT*sizeof(unsigned int);
#ifdef CACHECWLUT
    sm_size			= 2*NUM_SYMBOLS*sizeof(int) + block_size.x*sizeof(unsigned int)+ block_size.x*DPT*sizeof(unsigned int);
#endif
    if (sm_size>= MAX_SM_BLOCK_SIZE_GPU) {		printf("Ivalid kernel configuration: Reduce SM requirements!\n");	}
    else if (block_size.x>256)			 {		printf("Ivalid kernel configuration: Reduce number of threads/block!\n");	}
    else if (grid_size.x>32*1024)		 {		printf("Ivalid kernel configuration: Reduce number of blocks!\n");	}
    else {
        ktime			= 0;
        CUT_SAFE_CALL(cutResetTimer(timer));
        CUT_SAFE_CALL(cutStartTimer(timer));
        for (int i=0; i<NT; i++) {
            vlc_encode_kernel_dpt<<<grid_size, block_size, sm_size>>>(d_sourceData, d_codewords, d_codewordlens,  
#ifdef TESTING
                    d_cw32, d_cw32len, d_cw32idx, 
#endif
                    d_destData, d_cindex);//testedOK
        }
        cudaThreadSynchronize();
        CUT_CHECK_ERROR("Kernel execution failed\n");
        CUT_SAFE_CALL(cutStopTimer(timer));
        ktime += cutGetTimerValue(timer);
        //printf("Config. blocks: %d, threads: %d, dpt: %d, sm: %d [B]\n", grid_size.x, block_size.x, DPT, sm_size);
        printf("GPU Encoding time (DPT): %f (ms)\n", ktime/NT);
    }
    //////////////////* END KERNEL *///////////////////////////////////
#endif

#if 0
    //////////////////* DPT2 KERNEL *///////////////////////////////////
    block_size.x	= num_block_threads;
    grid_size.x		= num_blocks/DPT;
    sm_size			= block_size.x*sizeof(unsigned int)+ block_size.x*DPT*sizeof(unsigned int);
#ifdef CACHECWLUT
    sm_size			= 2*NUM_SYMBOLS*sizeof(int) + block_size.x*sizeof(unsigned int)+ block_size.x*DPT*sizeof(unsigned int);
#endif
    if (sm_size>= MAX_SM_BLOCK_SIZE_GPU) {		printf("Ivalid kernel configuration: Reduce SM requirements!\n");	}
    else if (block_size.x>256)			 {		printf("Ivalid kernel configuration: Reduce number of threads/block!\n");	}
    else if (grid_size.x>32*1024)		 {		printf("Ivalid kernel configuration: Reduce number of blocks!\n");	}
    else {
        ktime			= 0;
        CUT_SAFE_CALL(cutResetTimer(timer));
        CUT_SAFE_CALL(cutStartTimer(timer));
        for (int i=0; i<NT; i++) {
            vlc_encode_kernel_dpt2<<<grid_size, block_size, sm_size>>>(d_sourceData, d_codewords, d_codewordlens,  
#ifdef TESTING
                    d_cw32, d_cw32len, d_cw32idx, 
#endif
                    d_destData, d_cindex);//testedOK
        }
        cudaThreadSynchronize();
        CUT_CHECK_ERROR("Kernel execution failed\n");
        CUT_SAFE_CALL(cutStopTimer(timer));
        ktime += cutGetTimerValue(timer);
        //printf("Config. blocks: %d, threads: %d, dpt: %d, sm: %d [B]\n", grid_size.x, block_size.x, DPT, sm_size);
        printf("GPU Encoding time (DPT2): %f (ms)\n", ktime/NT);
    }
    //////////////////* END KERNEL *///////////////////////////////////
#endif





#if 0
    //////////////////* DPTT KERNEL *///////////////////////////////////
    grid_size.x		= num_blocks;
    block_size.x	= num_block_threads/DPT;
    sm_size			= 2*NUM_SYMBOLS*sizeof(int) + block_size.x*sizeof(unsigned int); 
#ifdef SMATOMICS
    sm_size+=block_size.x*DPT*sizeof(unsigned int);
#endif
    if (sm_size>= MAX_SM_BLOCK_SIZE_GPU) {		printf("Ivalid kernel configuration: Reduce SM requirements!\n");	}
    else if (block_size.x>256)			 {		printf("Ivalid kernel configuration: Reduce number of threads/block!\n");	}
    else if (grid_size.x>32*1024)		 {		printf("Ivalid kernel configuration: Reduce number of blocks!\n");	}
    else {
        ktime			= 0;
        CUT_SAFE_CALL(cutResetTimer(timer));
        CUT_SAFE_CALL(cutStartTimer(timer));
        for (int i=0; i<NT; i++) {
            vlc_encode_kernel_dptt<<<grid_size, block_size, sm_size>>>(d_sourceData, d_codewords, d_codewordlens,  
#ifdef TESTING
                    d_cw32, d_cw32len, d_cw32idx, 
#endif
                    d_destData, d_cindex);//testedOK
        }
        cudaThreadSynchronize();
        CUT_CHECK_ERROR("Kernel execution failed\n");
        CUT_SAFE_CALL(cutStopTimer(timer));
        ktime += cutGetTimerValue(timer);
        printf("GPU Encoding time (DPTT): %f (ms)\n", ktime/NT);
    }
    //////////////////* END KERNEL *///////////////////////////////////
#endif


#if 0 
    unsigned int num_scan_elements = grid_size.x;
    preallocBlockSums(num_scan_elements);
    cudaMemset(d_destDataPacked, 0, mem_size);
    printf("Num_blocks to be passed to scan is %d.\n", num_scan_elements);
    prescanArray(d_cindex2, d_cindex, num_scan_elements);

    //this works for kernels GM1, GM2, SM1, sm2
    //pack2<<<num_blocks/ENCODE_THREADS, ENCODE_THREADS>>>((unsigned int*)d_destData, d_cindex, d_cindex2, (unsigned int*)d_destDataPacked);
    //pack2<<<grid_size/16, 16>>>((unsigned int*)d_destData, d_cindex, d_cindex2, (unsigned int*)d_destDataPacked);

    pack2<<< num_scan_elements/16, 16>>>((unsigned int*)d_destData, d_cindex, d_cindex2, (unsigned int*)d_destDataPacked, num_elements/num_scan_elements);
    CUT_CHECK_ERROR("Pack2 Kernel execution failed\n");
    deallocBlockSums();

    CUDA_SAFE_CALL(cudaMemcpy(destData, d_destDataPacked, mem_size, cudaMemcpyDeviceToHost));
    compare_vectors((unsigned int*)crefData, (unsigned int*)destData, num_ints);
    //printdbg_data_bin("cpuout.txt", crefData, num_ints); 
    //printdbg_data_bin("gpuout_dpt2.txt", destData, num_ints); 
    //CUDA_SAFE_CALL(cudaMemcpy(cindex2, d_cindex2, num_blocks*sizeof(int), cudaMemcpyDeviceToHost));
    //printdbg_data_int("blockscan.txt", cindex2, num_blocks); 
    //CUDA_SAFE_CALL(cudaMemcpy(cw32, d_cw32, mem_size, cudaMemcpyDeviceToHost));
    //CUDA_SAFE_CALL(cudaMemcpy(cw32len, d_cw32len, mem_size, cudaMemcpyDeviceToHost));
    //CUDA_SAFE_CALL(cudaMemcpy(cw32idx, d_cw32idx, mem_size, cudaMemcpyDeviceToHost));
    //printdbg_gpu_data_detailed2("gpuout_dpt2detailed.txt", cw32, cw32len, cw32idx, num_ints);
#endif //ifdef TESTING

    // free(sourceData); 
    free(destData);  	free(codewords);  	free(codewordlens); free(crefData); 

    cudaFreeHost(sourceData);
    //    free(cw32);  free(cw32len); 
    CUDA_SAFE_CALL(cudaFree(d_sourceData)); 	CUDA_SAFE_CALL(cudaFree(d_destData)); CUDA_SAFE_CALL(cudaFree(d_destDataPacked));
    CUDA_SAFE_CALL(cudaFree(d_codewords)); 		CUDA_SAFE_CALL(cudaFree(d_codewordlens));
    //    CUDA_SAFE_CALL(cudaFree(d_cw32)); 		CUDA_SAFE_CALL(cudaFree(d_cw32len)); 	CUDA_SAFE_CALL(cudaFree(d_cw32idx)); 
    CUDA_SAFE_CALL(cudaFree(d_cindex)); CUDA_SAFE_CALL(cudaFree(d_cindex2));
    free(cindex2);

}



#if 0
//////////////////* DPT2 KERNEL *///////////////////////////////////
//grid_size.x		= num_blocks;
//block_size.x	= num_block_threads/DPT;
//sm_size			= 2*NUM_SYMBOLS*sizeof(int) + block_size.x*sizeof(unsigned int)+ block_size.x*DPT*sizeof(unsigned int);
block_size.x	= num_block_threads;
grid_size.x		= num_blocks/DPT;
sm_size			= 2*NUM_SYMBOLS*sizeof(int) + block_size.x*sizeof(unsigned int)+ block_size.x*DPT*sizeof(unsigned int);
if (sm_size>= MAX_SM_BLOCK_SIZE_GPU) {		printf("Ivalid kernel configuration: Reduce SM requirements!\n");	}
else if (block_size.x>256)			 {		printf("Ivalid kernel configuration: Reduce number of threads/block!\n");	}
else if (grid_size.x>32*1024)		 {		printf("Ivalid kernel configuration: Reduce number of blocks!\n");	}
else {
    ktime			= 0;
    CUT_SAFE_CALL(cutResetTimer(timer));
    CUT_SAFE_CALL(cutStartTimer(timer));
    for (int i=0; i<NT; i++) {
        vlc_encode_kernel_dpt2<<<grid_size, block_size, sm_size>>>(d_sourceData, d_codewords, d_codewordlens,  
#ifdef TESTING
                d_cw32, d_cw32len, d_cw32idx, 
#endif
                d_destData, d_cindex);//testedOK
    }
    cudaThreadSynchronize();
    CUT_CHECK_ERROR("Kernel execution failed\n");
    CUT_SAFE_CALL(cutStopTimer(timer));
    ktime += cutGetTimerValue(timer);
    //printf("Config. blocks: %d, threads: %d, dpt: %d, sm: %d [B]\n", grid_size.x, block_size.x, DPT, sm_size);
    printf("GPU Encoding time (DPT2): %f (ms)\n", ktime/NT);
}
//////////////////* END KERNEL *///////////////////////////////////
#endif
