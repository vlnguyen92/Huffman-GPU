
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//void cpu_vlc_encode_clean(unsigned int* indata, unsigned int num_elements, 
//					unsigned int* outdata, unsigned int *outsize, 
//					unsigned int *codewords, unsigned int* codewordlens) {
//
//  unsigned int *bitstreamPt = (unsigned int*) outdata;           
//  *bitstreamPt = 0x00000000U;
//  unsigned int startbit		= 0;
//  unsigned int totalBytes	= 0;
//
//  for(unsigned int k=0; k<num_elements; k++) {
//	unsigned int cw32		= 0;
//	unsigned int numbits	= 0;
//	unsigned int mask32		= 0;
//
//	for(unsigned int i=0; i<4;i++) {
//		unsigned char symbol = (unsigned char)(indata[k]>>(8*(3-i)));
//		cw32	= (cw32<<codewordlens[symbol]) | codewords[symbol];
//		numbits+=codewordlens[symbol];
//	}
//	while (numbits>0) {
//		int writebits =  min(32-startbit, numbits);
//		if (numbits==writebits)  mask32 = ( cw32&((1<<numbits)-1) )<<(32-startbit-numbits); //first make sure that the start of the word is clean, then shift to the left as many places as you need
//							else mask32 = cw32>>(numbits-writebits); //shift out the bits that can not fit
//      	*bitstreamPt = (*bitstreamPt) | mask32;
//		numbits = numbits - writebits;
//		startbit = (startbit+writebits)%32;
//		if (startbit == 0) { bitstreamPt++; *bitstreamPt = 0x00000000;  totalBytes += 4; }
//	}
//  }
//  totalBytes += (startbit/8) + ((startbit%8==0)? 0:1);
//  *outsize = totalBytes;
//}