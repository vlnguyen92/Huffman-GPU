#ifndef _CE_H_
#define _CE_H_

extern "C"
void cpu_vlc_encode(unsigned int* indata, unsigned int num_elements, 
					unsigned int* outdata, unsigned int *outsize, 
					uint256 *codewords, unsigned int* codewordlens);  
#endif


