
#include "stdafx.h"

#include "abitio.h"

extern "C"
void cpu_vlc_encode_lame(unsigned int* indata, unsigned int num_elements, 
					unsigned int* outdata, unsigned int *outsize, 
					unsigned int *codewords, unsigned int* codewordlens) {

  //BITIO_STRUCT bs;
BITIO_STRUCTI bsi;

  unsigned int count, cw32;
  unsigned char byte, cwlen;

  //open_bitstream_static(&bs, (unsigned char *)outdata);
	
init_bitstream_statici(&bsi, (unsigned int*)outdata, num_elements*sizeof(int));

  for(count=0; count<num_elements; count++) {
    byte =  (unsigned char ) (indata[count]>>24);
    cw32  = codewords[byte];
	cwlen = codewordlens[byte];
    //put_bits(cw32, cwlen, &bs, (unsigned char*)outdata);
	put_bitsi(&bsi, cw32, cwlen); 

	//put_bitsi_bbb(&bsi, cw32, cwlen); 


    byte =  (unsigned char ) (indata[count]>>16);
    cw32  = codewords[byte];
	cwlen = codewordlens[byte];
   // put_bits(cw32, cwlen, &bs, (unsigned char*)outdata);
	put_bitsi(&bsi, cw32, cwlen); 

	//put_bitsi_bbb(&bsi, cw32, cwlen); 


    byte =  (unsigned char )(indata[count]>>8);
    cw32  = codewords[byte];
	cwlen = codewordlens[byte];
   // put_bits(cw32, cwlen, &bs, (unsigned char*)outdata);
	put_bitsi(&bsi, cw32, cwlen); 

	//put_bitsi_bbb(&bsi, cw32, cwlen); 


	byte = (unsigned char ) indata[count];
    cw32  = codewords[byte];
	cwlen = codewordlens[byte];
    //put_bits(cw32, cwlen, &bs, (unsigned char*)outdata);
	put_bitsi(&bsi, cw32, cwlen); 

	//put_bitsi_bbb(&bsi, cw32, cwlen); 


  }

//  alignto8bits(&bs, (unsigned char *)outdata);

  *outsize = get_bitstream_len(&bsi)/8;
}
