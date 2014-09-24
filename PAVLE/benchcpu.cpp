#include "stdafx.h"

#include "cutil.h"
#include "print_helpers.h"
#include "comparison_helpers.h"
#include "cpuencode.h"

extern "C"
void runCPUEncodingTest(unsigned int num_blocks, unsigned int num_block_threads, unsigned int nnsymbols) { 
	uint num_elements = num_blocks * num_block_threads; 
	uint mem_size = num_elements * sizeof(int); 
	uint symbol_type_size = sizeof(int);
	unsigned int refbytesize;

	uint	*sourceData =	(uint*) malloc(mem_size);
	uint	*crefData   =	(uint*) malloc(mem_size);
	uint	*codewords	   = (uint*) malloc(NUM_SYMBOLS*symbol_type_size);
	uint	*codewordlens  = (uint*) malloc(NUM_SYMBOLS*symbol_type_size);

	for (int i=0; i<num_elements; i++) sourceData[i] = 0;
	for (int i=0; i<num_elements; i++) crefData[i] = 0;
	for (int i=0; i<NUM_SYMBOLS; i++) codewords[i] = 0;
	for (int i=0; i<NUM_SYMBOLS; i++) codewordlens[i] = 0;

	codewords[0]  = 0x00;	codewordlens[0] = 1;	
	codewords[1]  = 0x01;	codewordlens[1] = 2;
	codewords[2]  = 0x02;	codewordlens[2] = 2;
	codewords[3]  = 0x03;	codewordlens[3] = 2;
	codewords[4]  = 0x04;	codewordlens[4] =3;
	//printf("Codewords 32bit:\n");
	//print_array_in_hex(codewords, NUM_SYMBOLS);
	//print_array<uint>(codewordlens, NUM_SYMBOLS);
	for (int i=0; i<num_elements; i++) sourceData[i] = i%5; 

	unsigned int timer = 0;
	float ktime = 0.0f;
	printf("vlc_encode_kernel_cpu: \n");

	CUT_SAFE_CALL(cutCreateTimer(&timer));
	CUT_SAFE_CALL(cutStartTimer(timer));

	cpu_vlc_encode((unsigned int*)sourceData, num_elements, (unsigned int*)crefData,  &refbytesize, codewords, codewordlens);

	CUT_SAFE_CALL(cutStopTimer(timer));
	ktime = cutGetTimerValue(timer);
	CUT_SAFE_CALL(cutResetTimer(timer));

	printf("CPU Encoded to %d [B]\n", refbytesize);
	printf("Processing time: %f (ms)\n", ktime);
	CUT_SAFE_CALL(cutDeleteTimer(timer));

	//LogStats2("graph1","vlc_kernel_gm",ktime,(float)mem_size/1048576.0f);
	free(sourceData); 	free(codewords);  	free(codewordlens);  free(crefData);


}
