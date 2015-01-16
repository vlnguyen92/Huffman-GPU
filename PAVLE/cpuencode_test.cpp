#include "stdafx.h"
#include "print_helpers.h"

extern "C"
void cpu_vlc_encode(unsigned int* indata, unsigned int num_elements, 
					unsigned int* outdata, unsigned int *outsize, 
					unsigned int *codewords, unsigned int* codewordlens) {
  //bitstream init
  unsigned int *bitstreamPt = (unsigned int*) outdata;             /* Pointer to current byte   */
  *bitstreamPt = 0x00000000U;
  unsigned int startbit		= 0;
  unsigned int totalBytes	= 0;

  for(unsigned int k=0; k<num_elements; k++) {
	unsigned int cw32 = 0;
    unsigned int cwlen = 0;
	// unsigned int symbol; 
	//symbol = indata[i];
	//cw32 = symbol;
  	//numbits = 32;
	//cw32 = codewords[symbol];
	//numbits = codewordlens[symbol];
	unsigned int val32	= indata[k];
	unsigned int numbits = 0;
	unsigned int mask32;

		for(unsigned int i=0; i<4;i++) {
			unsigned char symbol = (unsigned char)(val32>>(8*(3-i)));
			unsigned int scw = codewords[symbol];
			unsigned int scwlen = codewordlens[symbol];
			printf("a[%d] = %d, byte %d = %d, cw: %d, cwlen: %d\n", k, indata[k], i, symbol, scw, scwlen);
			cw32	= (cw32<<codewordlens[symbol]) | codewords[symbol];
			numbits+=codewordlens[symbol];
		}
		printf("\nK:%d, cw32: %d, cwlen:%d\n", k, cw32, numbits);
		print32Bits(cw32); 


		while (numbits>0) {
			int writebits =  min(32-startbit, numbits);
			if (numbits==writebits)  mask32 = ( cw32&((1<<numbits)-1) )<<(32-startbit-numbits); //first make sure that the start of the word is clean, then shift to the left as many places as you need
								else mask32 = cw32>>(numbits-writebits); //shift out the bits that can not fit
			print32Bits(mask32); 
			*bitstreamPt = (*bitstreamPt) | mask32;
			print32Bits(*bitstreamPt); 
			numbits = numbits - writebits;
			startbit = (startbit+writebits)%32;
			if (startbit == 0) { /* start the next word */	bitstreamPt++; 	*bitstreamPt = 0x00000000;  totalBytes += 4; }
			print32BitsM(startbit); 
		}




  }

  printf("totalBytes: %d, startbit: %d\n", totalBytes, startbit);
  printf("Compressed to %d bits., \n", totalBytes*8+startbit);

  totalBytes += (startbit/8) + ((startbit%8==0)? 0:1);
  *outsize = totalBytes;
  printf("Padded data size %d [B].\n", totalBytes);
}
/////////////

inline void put_bits(unsigned int cw32, unsigned int numbits, BITIO_STRUCT *bsp) {
	unsigned int mask32;

	while (numbits>0) {
			int writebits =  min(32-bsp->startbit, numbits);
			if (numbits==writebits)  mask32 = ( cw32&((1<<numbits)-1) )<<(32-bsp->startbit-numbits); //first make sure that the start of the word is clean, then shift to the left as many places as you need
								else mask32 = cw32>>(numbits-writebits); //shift out the bits that can not fit
			*bsp->bitstreamPt = (*bsp->bitstreamPt) | mask32;
			numbits = numbits - writebits;
			bsp->startbit = (bsp->startbit+writebits)%32;
			if (bsp->startbit == 0) { bsp->bitstreamPt++; 	*bsp->bitstreamPt = 0x00000000;  totalBytes += 4; }
			print32BitsM(startbit); 
	}