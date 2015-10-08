#ifndef _LOADTESTDATA_H_
#define _LOADTESTDATA_H_

#include "testdatagen.h"

inline void initParams(char *file_name, uint num_block_threads, uint &num_blocks, uint &num_elements, uint &mem_size, uint symbol_type_size){
	if (file_name == NULL) {
		num_elements = num_blocks * num_block_threads;
		mem_size = num_elements * symbol_type_size;
	}
	else {
		FILE *f = fopen(file_name, "rb");
		if (!f) { perror(file_name); exit(1); }
		fseek(f, 0, SEEK_END);
		mem_size = ftell(f);
		fclose(f);
		num_elements = mem_size / symbol_type_size;
		//todo add check if we need 1 more block!
		num_blocks = num_elements / num_block_threads;
	}
}

inline void loadData(char *file_name, uint *sourceData, uint *codewords, uint *codewordlens, uint num_elements, uint mem_size, double &H){
	if (file_name == NULL) {
		//random code and data generation...
		//codewords[0] = 0x00;	codewordlens[0] = 1;
		//codewords[1] = 0x01;	codewordlens[1] = 2;
		//codewords[2] = 0x02;	codewordlens[2] = 2;
		//codewords[3] = 0x03;	codewordlens[3] = 2;
		//for (int i=0; i<num_elements; i++) sourceData[i] = i%4;
		////////////COPY THIS TO GENERATE CODEWORD TABLE AND SORUCE DATA /////////////////
		unsigned int num_symbols = NUM_SYMBOLS;
		generateCodewords(codewords, codewordlens, num_symbols);
		generateData(sourceData, num_elements, codewords, codewordlens, num_symbols);
		////generateSameBlocksOfData(sourceData, num_blocks, num_block_threads, codewords, codewordlens, NUM_SYMBOLS);
		////////////END COPY THIS TO GENERATE CODEWORD TABLE AND SORUCE DATA /////////////
	}
	else {
		/* LOAD SOURCE DATA AND MATCHING HUFFMAN CODES BY TJARK*/
		FILE *f = fopen(file_name, "rb");
		if (!f) { perror(file_name); exit(1); }
		fseek(f, 0, SEEK_SET);
		fread(sourceData, 1, mem_size, f);
		fclose(f);
		char buf[100];
		strcpy(buf, file_name);
		strcat(buf, "_codetable.txt");
		f = fopen(buf, "rt");
		if (!f) { perror(buf); exit(1); }
		uint symbol;
		while(fscanf(f, "%u", &symbol) != EOF) {
			fscanf(f, "%s", buf);
			codewordlens[symbol] = strlen(buf);
			for (unsigned int i=0; i<codewordlens[symbol]; i++)
				if (buf[i]-48 == 1)
					codewords[symbol] += (unsigned int)pow(2.0f, (int)(strlen(buf)-i-1));
		}
		fclose(f);

		unsigned int freqs[256]; //!!!! replaced NUM_SYMBOLS WITH 256, AS THE HUFFMAN CODING WORKS THAT WAY ONLY!!!!
		memset(freqs, 0, 256*sizeof(unsigned int));
		for (unsigned int i=0; i<mem_size; i++)
			freqs[((unsigned char*)sourceData)[i]]++;
		H = 0.0;
		for (unsigned int i=0; i<256; i++)
			if (freqs[i] > 0) {
				double p = (double)freqs[i] / (double)mem_size;
				H += p * log(p) / log(2.0);
			}
		H = -H;
		printf("\n%s, %u bytes, entropy %f\n\n", file_name, mem_size, H);

	}

}

#endif