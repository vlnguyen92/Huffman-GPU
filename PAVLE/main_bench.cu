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
#include "load_data.h"
//#include "test_kernels.cu"
#include "vlc_kernel_gm0.cu"
#include "vlc_kernel_gm1.cu"
#include "vlc_kernel_gm2.cu"
#include "vlc_kernel_sm1.cu"
#include "vlc_kernel_sm2.cu"
#include "vlc_kernel_sm2_huff.cu"
#include "vlc_kernel_dpt.cu"
#include "vlc_kernel_dpta.cu"
#include "vlc_kernel_dpt2.cu"
#include "vlc_kernel_dpt2b.cu"
#include "scan_kernel.cu"
#include "vlc_kernel_sm1ref.cu"

//void runVLCTest(unsigned int num_blocks, unsigned int num_block_threads, unsigned int nnsymbols = NUM_SYMBOLS);
void runVLCTest(char *file_name, uint num_block_threads, uint num_blocks=1);

extern "C" void cpu_vlc_encode(unsigned int* indata, unsigned int num_elements, unsigned int* outdata, unsigned int *outsize, unsigned int *codewords, unsigned int* codewordlens);
 
int main(int argc, char* argv[]){
	if(!InitCUDA()) { return 0;	}
	//unsigned int num_blocks			=  1; //1024; //2;// 32*1024; 	//2; //2048; //32*1024; //2048;//1024;	//2; //2; // 1;
	//unsigned int num_block_threads  = 256; //256; //16; //256;	//16; //128; //256; //128;// 256;	//16; //16;
	//runVLCTest(num_blocks, num_block_threads, 256);
	uint num_block_threads = 256;
	if (argc > 1)
		for (int i=1; i<argc; i++)
			runVLCTest(argv[i], num_block_threads);
	else {	runVLCTest(NULL, num_block_threads, 32*1024);	}
	//CUT_EXIT(argc, argv);
	CUDA_SAFE_CALL(cudaThreadExit());
	return 0;
}

//void runVLCTest(unsigned int num_blocks, unsigned int num_block_threads, unsigned int nnsymbols) { 
void runVLCTest(char *file_name, uint num_block_threads, uint num_blocks) {
	printf("CUDA! Starting VLC Tests!\n");
	uint num_elements; //uint num_elements = num_blocks * num_block_threads; 
	uint mem_size; //uint mem_size = num_elements * sizeof(int); 
	uint symbol_type_size = sizeof(int);
	//////// LOAD DATA ///////////////
	double H; // entropy
	initParams(file_name, num_block_threads, num_blocks, num_elements, mem_size, symbol_type_size);
	////////LOAD DATA ///////////////
	uint	*sourceData =	(uint*) malloc(mem_size);
	uint	*destData   =	(uint*) malloc(mem_size);
	uint	*crefData   =	(uint*) malloc(mem_size);

	uint	*codewords	   = (uint*) malloc(NUM_SYMBOLS*symbol_type_size);
	uint	*codewordlens  = (uint*) malloc(NUM_SYMBOLS*symbol_type_size);

	uint	*cw32 =		(uint*) malloc(mem_size);
	uint	*cw32len =	(uint*) malloc(mem_size);
	uint	*cw32idx =	(uint*) malloc(mem_size);

	memset(sourceData,   0, mem_size);
	memset(destData,     0, mem_size);
	memset(crefData,     0, mem_size);
	memset(cw32,         0, mem_size);
	memset(cw32len,      0, mem_size);
	memset(cw32idx,      0, mem_size);
	memset(codewords,    0, NUM_SYMBOLS*symbol_type_size);
	memset(codewordlens, 0, NUM_SYMBOLS*symbol_type_size);

	//////// LOAD DATA ///////////////
	loadData(file_name, sourceData, codewords, codewordlens, num_elements, mem_size, H);
	//printf("Codewords 32bit:\n");
	//print_array_in_hex(codewords, 256);
	//print_array<uint>(codewordlens, 256);
	//////// LOAD DATA ///////////////

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

	timer = 0; ktime = 0.0f;

	CUT_SAFE_CALL(cutCreateTimer(&timer));




	////for (int i=0; i<NT; i++) {



	//* CPU Encoding */
	ktime = 0;
	CUT_SAFE_CALL(cutResetTimer(timer));
	CUT_SAFE_CALL(cutStartTimer(timer));
	unsigned int refbytesize;
	cpu_vlc_encode((unsigned int*)sourceData, num_elements, (unsigned int*)crefData,  &refbytesize, codewords, codewordlens);
	CUT_SAFE_CALL(cutStopTimer(timer));
	ktime = cutGetTimerValue(timer);
	printf("CPU Encoding time (CPU): %f (ms)\n", ktime);
	printf("CPU Encoded to %d [B]\n", refbytesize);
	unsigned int num_ints = refbytesize/4 + ((refbytesize%4 ==0)?0:1);



	ktime = 0;
	CUT_SAFE_CALL(cutResetTimer(timer));
	CUT_SAFE_CALL(cutStartTimer(timer));
	vlc_encode_kernel_gm1<<<grid_size, block_size, sm_block_size_for_cwlens >>>(d_sourceData, d_codewords, d_codewordlens,  
																					#ifdef TESTING
																					d_cw32, d_cw32len, d_cw32idx, 
																					#endif
																				//d_destData); //testedOK2!
																				d_destData, d_cindex);
	cudaThreadSynchronize();
	CUT_CHECK_ERROR("Kernel execution failed\n");
	CUT_SAFE_CALL(cutStopTimer(timer));
	ktime = cutGetTimerValue(timer);
	printf("GPU Encoding time (GM1): %f (ms)\n", ktime);


	ktime = 0;
	CUT_SAFE_CALL(cutResetTimer(timer));
	CUT_SAFE_CALL(cutStartTimer(timer));
	///gm2: 2*NUM_SYMBOLS*sizeof(int) + sm_block_size_for_cwlens
	vlc_encode_kernel_gm2<<<grid_size, block_size, 2*NUM_SYMBOLS*sizeof(int) + sm_block_size_for_cwlens >>>(d_sourceData, d_codewords, d_codewordlens,  
																					#ifdef TESTING
																					d_cw32, d_cw32len, d_cw32idx, 
																					#endif
																							//d_destData); //testedOK2!
																				d_destData, d_cindex);
	cudaThreadSynchronize();
	CUT_CHECK_ERROR("Kernel execution failed\n");
	CUT_SAFE_CALL(cutStopTimer(timer));
	ktime = cutGetTimerValue(timer);
	printf("GPU Encoding time (GM2): %f (ms)\n", ktime);


	//ktime = 0;
	//CUT_SAFE_CALL(cutResetTimer(timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));
	//for (int i=0; i<NT; i++) {

	//vlc_encode_kernel_sm1ref<<<grid_size, block_size, 2*sm_block_size_for_cwlens >>>(d_sourceData, d_codewords, d_codewordlens,  
	//																				#ifdef TESTING
	//																				d_cw32, d_cw32len, d_cw32idx, 
	//																				#endif
	//																			d_destData); //testedOK2!
	//}
	//cudaThreadSynchronize();
	//CUT_CHECK_ERROR("Kernel execution failed\n");
	//CUT_SAFE_CALL(cutStopTimer(timer));
	//ktime += cutGetTimerValue(timer);
	//printf("GPU Encoding time (SM1REF): %f (ms)\n", ktime/NT);



	ktime = 0;
	CUT_SAFE_CALL(cutResetTimer(timer));
	CUT_SAFE_CALL(cutStartTimer(timer));
		for (int i=0; i<NT; i++) {
	vlc_encode_kernel_sm1<<<grid_size, block_size, 2*sm_block_size_for_cwlens >>>(d_sourceData, d_codewords, d_codewordlens,  
																					#ifdef TESTING
																					d_cw32, d_cw32len, d_cw32idx, 
																					#endif
																				//d_destData); //testedOK2!
																				d_destData, d_cindex);

		}
	cudaThreadSynchronize();
	CUT_CHECK_ERROR("Kernel execution failed\n");
	CUT_SAFE_CALL(cutStopTimer(timer));
	ktime += cutGetTimerValue(timer);
	printf("GPU Encoding time (SM1): %f (ms)\n", ktime/NT);


	ktime = 0;
	CUT_SAFE_CALL(cutResetTimer(timer));
	CUT_SAFE_CALL(cutStartTimer(timer));
		for (int i=0; i<NT; i++) {
	vlc_encode_kernel_sm2<<<grid_size, block_size, 2*NUM_SYMBOLS*sizeof(int) + sm_block_size_for_cwlens>>>(d_sourceData, d_codewords, d_codewordlens,  
																					#ifdef TESTING
																					d_cw32, d_cw32len, d_cw32idx, 
																					#endif
																				//d_destData); //testedOK2!
																				d_destData, d_cindex);
		}
	cudaThreadSynchronize();
	CUT_CHECK_ERROR("Kernel execution failed\n");
	CUT_SAFE_CALL(cutStopTimer(timer));
	ktime += cutGetTimerValue(timer);
	printf("GPU Encoding time (SM2): %f (ms)\n", ktime/NT);


	ktime = 0;
	CUT_SAFE_CALL(cutResetTimer(timer));
	CUT_SAFE_CALL(cutStartTimer(timer));
		for (int i=0; i<NT; i++) {
	vlc_encode_kernel_sm2_huff<<<grid_size, block_size, 2*NUM_SYMBOLS*sizeof(int) + sm_block_size_for_cwlens>>>(d_sourceData, d_codewords, d_codewordlens,  
																					#ifdef TESTING
																					d_cw32, d_cw32len, d_cw32idx, 
																					#endif
																				//d_destData); //testedOK2!
																				d_destData, d_cindex);
		}
	cudaThreadSynchronize();
	CUT_CHECK_ERROR("Kernel execution failed\n");
	CUT_SAFE_CALL(cutStopTimer(timer));
	ktime += cutGetTimerValue(timer);
	printf("GPU Encoding time (SM2HUFF): %f (ms)\n", ktime/NT);


	ktime = 0;
	CUT_SAFE_CALL(cutResetTimer(timer));
	CUT_SAFE_CALL(cutStartTimer(timer));
		for (int i=0; i<NT; i++) {
	vlc_encode_kernel_dpt<<<num_blocks, num_block_threads/DPT, (ENCODE_THREADS+2*NUM_SYMBOLS)*sizeof(unsigned int)>>>
																		(d_sourceData, d_codewords, d_codewordlens,  
																					#ifdef TESTING
																					d_cw32, d_cw32len, d_cw32idx, 
																					#endif
																					d_destData, d_cindex);//testedOK
		}
	cudaThreadSynchronize();
	CUT_CHECK_ERROR("Kernel execution failed\n");
	CUT_SAFE_CALL(cutStopTimer(timer));
	ktime += cutGetTimerValue(timer);
	printf("GPU Encoding time (DPT): %f (ms)\n", ktime/NT);

	ktime = 0;
	CUT_SAFE_CALL(cutResetTimer(timer));
	CUT_SAFE_CALL(cutStartTimer(timer));
		for (int i=0; i<NT; i++) {
	////(2*NUM_SYMBOLS + ENCODE_THREADS + ENCODE_THREADS*DPT)*sizeof(unsigned int) 
	vlc_encode_kernel_dpt2<<<num_blocks, num_block_threads/DPT, (2*NUM_SYMBOLS + ENCODE_THREADS + ENCODE_THREADS*DPT)*sizeof(unsigned int) >>>
																		(d_sourceData, d_codewords, d_codewordlens,  
																					#ifdef TESTING
																					d_cw32, d_cw32len, d_cw32idx, 
																					#endif
																					d_destData, d_cindex);//testedOK
		}
	cudaThreadSynchronize();
	CUT_CHECK_ERROR("Kernel execution failed\n");
	CUT_SAFE_CALL(cutStopTimer(timer));
	ktime += cutGetTimerValue(timer);
	printf("GPU Encoding time (DPT2): %f (ms)\n", ktime/NT);

	ktime = 0;
	CUT_SAFE_CALL(cutResetTimer(timer));
	CUT_SAFE_CALL(cutStartTimer(timer));
		for (int i=0; i<NT; i++) {
	scan_kernel<<<grid_size, block_size,sm_block_size_for_cwlens>>>(d_cw32len, d_cw32idx);
		}
	cudaThreadSynchronize();
	CUT_CHECK_ERROR("Kernel execution failed\n");
	CUT_SAFE_CALL(cutStopTimer(timer));
	ktime+= cutGetTimerValue(timer);
	printf("GPU Encoding time (SCAN): %f (ms)\n", ktime/NT);




#ifdef TESTING
	CUDA_SAFE_CALL(cudaMemcpy(cw32, d_cw32, mem_size, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(cw32len, d_cw32len, mem_size, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(cw32idx, d_cw32idx, mem_size, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(destData, d_destData, mem_size, cudaMemcpyDeviceToHost));
	//if (num_blocks>1) { //check just a print out  just the first coded block! //after bit-tigtht packing is added we can remove this
	//	num_ints = cw32idx[0]/32 + (cw32idx[0]%32==0)?0:1; 
	//	num_elements = num_ints;
	//}
	//printdbg_gpu_data_detailed2("gpuout02.txt", cw32, cw32len, cw32idx, num_elements);
	printdbg_data_bin("cpuout.txt", crefData, num_ints); 
	printdbg_data_bin("gpuout.txt", destData, num_ints); 
	compare_vectors((unsigned int*)crefData, (unsigned int*)destData, num_ints);
#endif //ifdef TESTING

	//LogStats2("graph1","vlc_kernel_gm",ktime,(float)mem_size/1048576.0f);

	free(sourceData); free(destData);  	free(codewords);  	free(codewordlens); free(cw32);  free(cw32len); free(crefData); 
	CUDA_SAFE_CALL(cudaFree(d_sourceData)); 	CUDA_SAFE_CALL(cudaFree(d_destData));
	CUDA_SAFE_CALL(cudaFree(d_codewords)); 		CUDA_SAFE_CALL(cudaFree(d_codewordlens));
	CUDA_SAFE_CALL(cudaFree(d_cw32)); 		CUDA_SAFE_CALL(cudaFree(d_cw32len)); 	CUDA_SAFE_CALL(cudaFree(d_cw32idx)); 
	CUDA_SAFE_CALL(cudaFree(d_cindex));
	
}



	//ktime = 0;
	//CUT_SAFE_CALL(cutResetTimer(timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));
	//vlc_encode_kernel_gm0<<<grid_size, block_size, sm_block_size_for_cwlens >>>(d_sourceData, d_codewords, d_codewordlens,  
	//																				#ifdef TESTING
	//																				d_cw32, d_cw32len, d_cw32idx, 
	//																				#endif
	//																			d_destData);  //testedOK2!



	//cudaThreadSynchronize();
	//CUT_CHECK_ERROR("Kernel execution failed\n");
	//CUT_SAFE_CALL(cutStopTimer(timer));
	//ktime = cutGetTimerValue(timer);
	//printf("GPU Encoding time: %f (ms)\n", ktime);


	//ktime = 0;
	//CUT_SAFE_CALL(cutResetTimer(timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));
	//	for (int i=0; i<NT; i++) {
	//vlc_encode_kernel_dpta<<<num_blocks, num_block_threads/DPT, (ENCODE_THREADS+2*NUM_SYMBOLS)*sizeof(unsigned int)>>>
	//																	(d_sourceData, d_codewords, d_codewordlens,  
	//																				#ifdef TESTING
	//																				d_cw32, d_cw32len, d_cw32idx, 
	//																				#endif
	//																			d_destData, d_cindex);//testedOK2
	//	}
	//cudaThreadSynchronize();
	//CUT_CHECK_ERROR("Kernel execution failed\n");
	//CUT_SAFE_CALL(cutStopTimer(timer));
	//ktime += cutGetTimerValue(timer);
	//printf("GPU Encoding time (DPTa): %f (ms)\n", ktime/NT);

	//ktime = 0;
	//CUT_SAFE_CALL(cutResetTimer(timer));
	//CUT_SAFE_CALL(cutStartTimer(timer));
	//	for (int i=0; i<NT; i++) {
	////(2*num_symbols + encode_threads + 2*encode_threads*dpt)*sizeof(unsigned int) 
	//vlc_encode_kernel_dpt2b<<<num_blocks, num_block_threads/DPT, (2*NUM_SYMBOLS + ENCODE_THREADS + 2*ENCODE_THREADS*DPT)*sizeof(unsigned int) >>>
	//																	(d_sourceData, d_codewords, d_codewordlens,  
	//																				#ifdef TESTING
	//																				d_cw32, d_cw32len, d_cw32idx, 
	//																				#endif
	//																				d_destData, d_cindex);
	//	}
	//cudaThreadSynchronize();
	//CUT_CHECK_ERROR("Kernel execution failed\n");
	//CUT_SAFE_CALL(cutStopTimer(timer));
	//ktime += cutGetTimerValue(timer);
	//printf("GPU Encoding time (DPT2b): %f (ms)\n", ktime/NT);

