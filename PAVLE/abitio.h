/*
 * Copyright Ana Balevic, 2009. All rights reserved.
 */

#ifndef _ABITIO_H_
#define _ABITIO_H_

#include "stdafx.h"

struct BITIO_STRUCTI {
  unsigned int *currentidx;             /* Pointer to current byte   */
  unsigned int bitpos;
  unsigned int total_bit_count;
};

inline void init_bitstream_statici(BITIO_STRUCTI *bsp, unsigned int *bitstream, unsigned int buff_mem_size) {
  memset(bitstream, 0, buff_mem_size);
  bsp->currentidx	= bitstream;	    /* Point to the beginning of bitstream buffer				*/
  bsp->bitpos = 0;
  bsp->total_bit_count	= 0;				/* Number of bits in bistream buffer - set by put_bit, read by get_bit */
}

inline void put_bitsi(BITIO_STRUCTI *bsp, unsigned int data32, unsigned int num_bits) {
	bsp->total_bit_count+= num_bits;

	if (bsp->bitpos + num_bits >32) {
		num_bits = bsp->bitpos + num_bits - 32;
		*bsp->currentidx |= (data32>>num_bits); // get the first few bits that fit
		bsp->bitpos = 0;
		bsp->currentidx++;
	}
	*bsp->currentidx |= (data32<<(32-bsp->bitpos-num_bits));
	bsp->bitpos+=num_bits;

}

//put bit by bit; just for checking if the result is ok...
inline void put_bitsi_bbb(BITIO_STRUCTI *bsp, unsigned int data32, unsigned int num_bits) {
	unsigned int mask32 = 1<<(num_bits-1);
	unsigned int bit = 0;
	while (mask32) {
		bit = (mask32 & data32)? 1:0;
		if (bsp->bitpos == 32) {
			bsp->currentidx++;
			bsp->bitpos = 0;
		}
		*bsp->currentidx |= (bit<<(32-bsp->bitpos-1)); //sets only 1 bit always...

		bsp->bitpos++;
		bsp->total_bit_count++;
		mask32>>=1;
	}
}

inline unsigned int get_bitstream_len(BITIO_STRUCTI *bsp) {
  return bsp->total_bit_count;
}
#endif //#ifndef _ABITIO_H_
