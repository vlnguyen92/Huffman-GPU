#ifndef _CEL_H_
#define _CEL_H_

extern "C"
void cpu_vlc_encode_lame(unsigned int* indata, unsigned int num_elements, 
					unsigned int* destData, unsigned int *outsize, 
					unsigned int *codewords, unsigned int* codewordlens);  
#endif


