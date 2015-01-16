#include "stdafx.h"

#include "print_helpers.h"
#include "cpuencode.h"
#include <iostream>

using namespace std;


#if 0

// The max. codeword length for each byte symbol is 32-bits

extern "C"
void cpu_vlc_encode(unsigned int* indata, unsigned int num_elements, 
                    unsigned int* outdata, unsigned int *outsize, 
                    unsigned int *codewords, unsigned int* codewordlens) {
    unsigned int *bitstreamPt = (unsigned int*) outdata;             /* Pointer to current byte   */
    *bitstreamPt = 0x00000000U;
    unsigned int startbit		= 0;
    unsigned int totalBytes	= 0;

    for(unsigned int k=0; k<num_elements; k++) {
        unsigned int cw32 = 0;
        unsigned int val32	= indata[k];
        unsigned int numbits = 0;
        unsigned int mask32;

        for(unsigned int i=0; i<4;i++) {
            unsigned char symbol = (unsigned char)(val32>>(8*(3-i))); cw32	= codewords[symbol];
            numbits = codewordlens[symbol];

            while (numbits>0) {
                int writebits =  min(32-startbit, numbits);
                if (numbits==writebits)  mask32 = ( cw32&((1<<numbits)-1) )<<(32-startbit-numbits); //first make sure that the start of the word is clean, then shift to the left as many places as you need
                else mask32 = cw32>>(numbits-writebits); //shift out the bits that can not fit
                *bitstreamPt = (*bitstreamPt) | mask32;
                numbits = numbits - writebits;
                startbit = (startbit+writebits)%32;
                if (startbit == 0) { bitstreamPt++;  *bitstreamPt = 0x00000000;  totalBytes += 4; }
            }
        }

    }
    totalBytes += (startbit/8) + ((startbit%8==0)? 0:1); //return aligned to 8-bits
    *outsize = totalBytes;
}

//////////////////////////////////////////////////////////////////////
/// ALTERNATIVE CODER
/// ASSUMPTION: The max. length of 4 combined codewords can be 2x original data, i.e. g 64 bits
///////////////////////////////////////////////////////////////////////

#else

extern "C"
void cpu_vlc_encode(unsigned int* indata, unsigned int num_elements, 
                    unsigned int* outdata, unsigned int *outsize, 
                    uint256 *codewords, unsigned int* codewordlens) {
    unsigned int *bitstreamPt = (unsigned int*) outdata;             /* Pointer to current byte   */
    //    cout << "I just got called" << endl;
    //assume memset is done.
    *bitstreamPt = 0x00000000U;
    unsigned int startbit		= 0;
    unsigned int totalBytes	= 0;

    for(unsigned int k=0; k<num_elements; k++) {
        uint256 cw256 = 0, mask256=0;
        uint256 one = 1, num;
        unsigned int val32	= indata[k];
        unsigned int numbits = 0;
        unsigned int mask32, temp32;


        for(unsigned int i=0; i<4;i++) {
            unsigned char symbol = (unsigned char)(val32>>(8*(3-i)));
            cw256	= (cw256<<codewordlens[symbol]) | codewords[symbol];
            numbits+=codewordlens[symbol];
            //if (numbits>32) printf("WARRNING! Element %d is combined into numbits = %d!!!!!!!\n", k, numbits);
        }

        while (numbits>0) {
            int writebits =  min(32-startbit, numbits);
            if (numbits==writebits)  {
                temp32 = cw256.operator=(cw256); //(cw256 & 0xFFFFFFFF); 
                mask32 = temp32<<(32-startbit-numbits);
            }
            else {
                num = (cw256>>(numbits-writebits));
                mask32 = num.operator=(num);               
                cw256 = cw256 & ((one<<(numbits-writebits))-1);
            }
            *bitstreamPt = (*bitstreamPt) | mask32;
            numbits = numbits - writebits;
            startbit = (startbit+writebits)%32;
            if (startbit == 0) { 
                bitstreamPt++;  
                *bitstreamPt = 0x00000000; 
                totalBytes += 4;
            }
        }
    }
    totalBytes += (startbit/8) + ((startbit%8==0)? 0:1); //return aligned to 8-bits
    *outsize = totalBytes;
}
#endif
