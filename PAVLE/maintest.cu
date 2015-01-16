#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cutil.h>
#include <cuda_runtime.h>
#include "cuda_helpers.h"
#include "print_helpers.h"
#include "comparison_helpers.h"
#include "stats_logger.h"
#include "testdatagen.h"

#include "test_kernels.cu"
#include "vlc_kernel_gm0.cu"
#include "vlc_kernel_gm1.cu"
#include "vlc_kernel_gm2.cu"
#include "vlc_kernel_sm1.cu"
#include "vlc_kernel_sm4.cu"

#include "vlc_kernel_dpt.cu"
#include "vlc_kernel_dpta.cu"
#include "vlc_kernel_dpt2.cu"




void runVLCTest(unsigned int num_blocks, unsigned int num_block_threads, unsigned int nnsymbols = NUM_SYMBOLS);

extern "C" void cpu_vlc_encode(unsigned int* indata, unsigned int num_elements, unsigned int* outdata, unsigned int *outsize, unsigned int *codewords, unsigned int* codewordlens);
 
int main(int argc, char* argv[]){
	if(!InitCUDA()) { return 0;	}
	unsigned int num_blocks			=  1; //1024; //2;// 32*1024; 	//2; //2048; //32*1024; //2048;//1024;	//2; //2; // 1;
	unsigned int num_block_threads  = 256; //256; //16; //256;	//16; //128; //256; //128;// 256;	//16; //16;
	runVLCTest(num_blocks, num_block_threads, 256);
	CUT_EXIT(argc, argv);
	return 0;
}

void runVLCTest(unsigned int num_blocks, unsigned int num_block_threads, unsigned int nnsymbols) { 
	uint num_elements = num_blocks * num_block_threads; 
	uint mem_size = num_elements * sizeof(int); 
	uint symbol_type_size = sizeof(int);

	uint	*sourceData =	(uint*) malloc(mem_size);
	uint	*destData   =	(uint*) malloc(mem_size);
	uint	*crefData   =	(uint*) malloc(mem_size);

	uint	*codewords	   = (uint*) malloc(NUM_SYMBOLS*symbol_type_size);
	uint	*codewordlens  = (uint*) malloc(NUM_SYMBOLS*symbol_type_size);

	uint	*cw32 =		(uint*) malloc(mem_size);
	uint	*cw32len =	(uint*) malloc(mem_size);
	uint	*cw32idx =	(uint*) malloc(mem_size);

	//TODO; memset doesn't work correctly...
	//memset(sourceData, num_elements*sizeof(int), 0); 
	//memset(destData, num_elements*sizeof(int), 0);
	//memset(crefData, num_elements*sizeof(int), 0);
	//memset(codewords, NUM_SYMBOLS*sizeof(int), 0);
	//memset(codewordlens, NUM_SYMBOLS*sizeof(int), 0);
	//memset(cw32, num_elements*sizeof(int), 0);
	//memset(cw32len, num_elements*sizeof(int), 0);
	//memset(cw32idx, num_elements*sizeof(int), 0);

for (int i=0; i<num_elements; i++) sourceData[i] = 0;
for (int i=0; i<num_elements; i++) destData[i] = 0;
for (int i=0; i<num_elements; i++) crefData[i] = 0;
for (int i=0; i<num_elements; i++) cw32[i] = 0;
for (int i=0; i<num_elements; i++) cw32len[i] = 0;
for (int i=0; i<num_elements; i++) cw32idx[i] = 0;
for (int i=0; i<NUM_SYMBOLS; i++) codewords[i] = 0;
for (int i=0; i<NUM_SYMBOLS; i++) codewordlens[i] = 0;

//////////////////////////////////////////////////////////////
/// TODO: replace with loading the file and the codes
/// loadData(sourceData, ...)
/// assignCodes(...)
/////////////////////////////////////////////////////////////
	codewords[0]  = 0x00;	codewordlens[0] = 1;	
	codewords[1]  = 0x01;	codewordlens[1] = 2;
	codewords[2]  = 0x02;	codewordlens[2] = 2;
	codewords[3]  = 0x03;	codewordlens[3] = 2;
	//codewords[4]  = 0x04;	codewordlens[4] =3;
	printf("Codewords 32bit:\n");
	print_array_in_hex(codewords, NUM_SYMBOLS);
	print_array<uint>(codewordlens, NUM_SYMBOLS);
	for (int i=0; i<num_elements; i++) sourceData[i] = i%4; //sourceData[i] = i%5; 
	printf("Source data:\n");
	print_array(sourceData, num_elements);
	printf("Ouput data:\n");
	print_array(destData, num_elements);
/////////////////////////////////////////////////////////

	//////////COPY THIS TO GENERATE CODEWORD TABLE AND SORUCE DATA /////////////////
	//unsigned int num_symbols = nnsymbols;
	//generateCodewords(codewords, codewordlens, num_symbols);
	//generateData(sourceData, num_elements, codewords, codewordlens, num_symbols);
	////generateSameBlocksOfData(sourceData, num_blocks, num_block_threads, codewords, codewordlens, num_symbols);
	//////////END COPY THIS TO GENERATE CODEWORD TABLE AND SORUCE DATA /////////////////

	unsigned int	*d_sourceData, *d_destData;
	unsigned int	*d_codewords, *d_codewordlens;
	unsigned int	*d_cw32, *d_cw32len, *d_cw32idx, *d_cindex;

	CUDA_SAFE_CALL(cudaMalloc((void**) &d_sourceData,		  mem_size));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_destData,			  mem_size));

	CUDA_SAFE_CALL(cudaMalloc((void**) &d_codewords,		  NUM_SYMBOLS*symbol_type_size));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_codewordlens,		  NUM_SYMBOLS*symbol_type_size));

	CUDA_SAFE_CALL(cudaMalloc((void**) &d_cw32,				  mem_size));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_cw32len,			  mem_size));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_cw32idx,			  mem_size));

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_cindex,        num_blocks*sizeof(unsigned int)));

	CUDA_SAFE_CALL(cudaMemcpy(d_sourceData,		sourceData,		mem_size,		cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_codewords,		codewords,		NUM_SYMBOLS*symbol_type_size,	cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_codewordlens,	codewordlens,	NUM_SYMBOLS*symbol_type_size,	cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_destData,		destData,		mem_size,		cudaMemcpyHostToDevice));

    dim3 grid_size(num_blocks,1,1);
    dim3 block_size(num_block_threads, 1, 1);
    unsigned int sm_block_size_for_cwlens = num_block_threads * sizeof(unsigned int);

	unsigned int timer = 0;
	float ktime = 0.0f;
	unsigned int NT = 1; //number of runs for each execution time


	printf("vlc_encode_kernel_gm1: gm atomics, CodewordLUTs - ints, optimized\n");
	timer = 0; ktime = 0.0f;

	CUT_SAFE_CALL(cutCreateTimer(&timer));
	for (int i=0; i<NT; i++) {
	CUT_SAFE_CALL(cutStartTimer(timer));

	//test_copy<<<grid_size, block_size>>>(d_sourceData, 	d_destData); //testedOK
	//check_gm_kernel<<<grid_size, block_size>>>(d_destData);


	vlc_encode_kernel_gm0<<<grid_size, block_size, sm_block_size_for_cwlens >>>(d_sourceData, d_codewords, d_codewordlens,  
																					#ifdef TESTING
																					d_cw32, d_cw32len, d_cw32idx, 
																					#endif
																				d_destData);  //testedOK2!

	//vlc_encode_kernel_gm1<<<grid_size, block_size, sm_block_size_for_cwlens >>>(d_sourceData, d_codewords, d_codewordlens,  
	//																				#ifdef TESTING
	//																				d_cw32, d_cw32len, d_cw32idx, 
	//																				#endif
	//																			d_destData); //testedOK2!

	///gm2: 2*NUM_SYMBOLS*sizeof(int) + sm_block_size_for_cwlens

	//vlc_encode_kernel_gm2<<<grid_size, block_size, 2*NUM_SYMBOLS*sizeof(int) + sm_block_size_for_cwlens >>>(d_sourceData, d_codewords, d_codewordlens,  
	//																				#ifdef TESTING
	//																				d_cw32, d_cw32len, d_cw32idx, 
	//																				#endif
	//																			d_destData); //testedOK2!

	//vlc_encode_kernel_dpt<<<num_blocks, num_block_threads/DPT, (ENCODE_THREADS+2*NUM_SYMBOLS)*sizeof(unsigned int)>>>
	//																	(d_sourceData, d_codewords, d_codewordlens,  
	//																				#ifdef TESTING
	//																				d_cw32, d_cw32len, d_cw32idx, 
	//																				#endif
	//																				d_destData, d_cindex);//testedOK



	//////(2*NUM_SYMBOLS + ENCODE_THREADS + ENCODE_THREADS*DPT)*sizeof(unsigned int) 
	//vlc_encode_kernel_dpt2<<<num_blocks, num_block_threads/DPT, (2*NUM_SYMBOLS + ENCODE_THREADS + ENCODE_THREADS*DPT)*sizeof(unsigned int) >>>
	//																	(d_sourceData, d_codewords, d_codewordlens,  
	//																				#ifdef TESTING
	//																				d_cw32, d_cw32len, d_cw32idx, 
	//																				#endif
	//																				d_destData, d_cindex);//testedOK

	//vlc_encode_kernel_dpta<<<num_blocks, num_block_threads/DPT, (ENCODE_THREADS+2*NUM_SYMBOLS)*sizeof(unsigned int)>>>
	//																	(d_sourceData, d_codewords, d_codewordlens,  
	//																				#ifdef TESTING
	//																				d_cw32, d_cw32len, d_cw32idx, 
	//																				#endif
	//																			d_destData, d_cindex);//testedOK2


//vlc_encode_kernel_sm1<<<grid_size, block_size, 2*sm_block_size_for_cwlens >>>(d_sourceData, d_codewords, d_codewordlens,  
//																					#ifdef TESTING
//																					d_cw32, d_cw32len, d_cw32idx, 
//																					#endif
//																				d_destData); //testedOK2!

	//vlc_encode_kernel_sm4<<<grid_size, block_size, 2*NUM_SYMBOLS*sizeof(int) + sm_block_size_for_cwlens>>>(d_sourceData, d_codewords, d_codewordlens,  
	//																				#ifdef TESTING
	//																				d_cw32, d_cw32len, d_cw32idx, 
	//																				#endif
	//															d_destData); //testedOK2

	cudaThreadSynchronize();
	CUT_CHECK_ERROR("Kernel execution failed\n");
	CUT_SAFE_CALL(cutStopTimer(timer));
	ktime += cutGetTimerValue(timer);
	CUT_SAFE_CALL(cutResetTimer(timer));
	}
	ktime /= NT;
	printf("Processing time: %f (ms)\n", ktime);
	CUT_SAFE_CALL(cutDeleteTimer(timer));


#ifdef TESTING
	CUDA_SAFE_CALL(cudaMemcpy(cw32, d_cw32, mem_size, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(cw32len, d_cw32len, mem_size, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(cw32idx, d_cw32idx, mem_size, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(destData, d_destData, mem_size, cudaMemcpyDeviceToHost));
	printf("Source data:\n");
	print_array(sourceData, num_elements);
	//printf("Ouput data:\n");
	//print_array(destData, num_elements);
    //compare_vectors(sourceData, destData, num_elements);
	//compare_vectors(sourceData, cw32, num_elements);

	printf("cw32:\n");
	print_array(cw32, num_elements);
	printf("cw32len:\n");
	print_array(cw32len, num_elements);
	printf("cw32idx:\n");
	print_array(cw32idx, num_elements);
	printdbg_gpu_data_detailed2("gpuout02.txt", cw32, cw32len, cw32idx, num_elements);
	printf("Ouput data:\n");
	print_array(destData, num_elements);
    //compare_vectors(sourceData, destData, num_elements);

	unsigned int refbytesize;
	cpu_vlc_encode((unsigned int*)sourceData, num_elements, (unsigned int*)crefData,  &refbytesize, codewords, codewordlens);
	printf("CPU Encoded to %d [B]\n", refbytesize);
	unsigned int num_ints = refbytesize/4 + ((refbytesize%4 ==0)?0:1);

	//FILE *cpudump	= fopen("cpuout1.txt", "wt"); //"wb");
	////fwrite((unsigned char*)crefData, sizeof(char), refbytesize, cpudump);
	//for(unsigned int i=0; i< num_ints; i++) {
	//	unsigned int mask = 0x80000000;
	//	for(unsigned int j = 0; j < 32; j++)  {
	//	if (crefData[i] & mask) fprintf(cpudump, "1"); //printf("1");
	//		else  fprintf(cpudump, "0");//printf("0");
	//	mask = mask >> 1;
	//	}
	//	fprintf(cpudump, ".\n");
	//}
	//fclose(cpudump);

	printdbg_data_bin("cpuout11.txt", crefData, num_ints); 

	printdbg_data_bin("gpuout11.txt", destData, num_ints); 

	//FILE *gpudump	= fopen("gpuout1.txt", "wt"); //"wb");
	////fwrite((unsigned char*)crefData, sizeof(char), refbytesize, cpudump);
	//for(unsigned int i=0; i< num_ints; i++) {
	//	unsigned int mask = 0x80000000;
	//	for(unsigned int j = 0; j < 32; j++)  {
	//	if (destData[i] & mask) fprintf(gpudump, "1"); //printf("1");
	//		else  fprintf(gpudump, "0");//printf("0");
	//	mask = mask >> 1;
	//	}
	//	fprintf(gpudump, ".\n");
	//}
	//fclose(gpudump);


	//printf("CPU ENCODING:\n");
	//print_array_ints_as_bits((unsigned int*)crefData,  num_ints);
	//print_array((unsigned int*)crefData,  num_ints);
	//printf("GPU ENCODING:\n");
	//print_array_ints_as_bits((unsigned int*)destData,  num_ints);
	//print_array((unsigned int*)destData,  num_ints);
	//printf("CPU encoded data:\n");
	//print_array(crefData, num_elements);
	//compare_vectors((unsigned char*)crefData, (unsigned char*)destData, refbytesize);
	//print_compare_array_ints_as_bits((unsigned int*)crefData, (unsigned int*)destData, num_ints);
	//compare_vectors(sourceData, crefData, num_elements);
	compare_vectors((unsigned int*)crefData, (unsigned int*)destData, num_ints);
#endif //ifdef TESTING


	//LogStats2("graph1","vlc_kernel_gm",ktime,(float)mem_size/1048576.0f);


	free(sourceData); free(destData);  	free(codewords);  	free(codewordlens); free(cw32);  free(cw32len); free(crefData); 
	CUDA_SAFE_CALL(cudaFree(d_sourceData)); 	CUDA_SAFE_CALL(cudaFree(d_destData));
	CUDA_SAFE_CALL(cudaFree(d_codewords)); 		CUDA_SAFE_CALL(cudaFree(d_codewordlens));
	CUDA_SAFE_CALL(cudaFree(d_cw32)); 		CUDA_SAFE_CALL(cudaFree(d_cw32len)); 	CUDA_SAFE_CALL(cudaFree(d_cw32idx)); 
	CUDA_SAFE_CALL(cudaFree(d_cindex));
	
}
