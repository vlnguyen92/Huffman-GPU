/*  Copyright (c) 2014 Linh Nguyen.
    Permission is granted to copy, distribute and/or modify this document
    under the terms of the GNU Free Documentation License, Version 1.2
    or any later version published by the Free Software Foundation;
    with no Invariant Sections, no Front-Cover Texts, and no Back-Cover
    Texts.  A copy of the license is included in the section entitled "GNU
    Free Documentation License".*/
#include "huffTree.h"
#include "hist.cu"

int main(int argc, char** argv)
{
    // Build frequency table
    //    string filename;
    //    cout << "Enter file name" << endl;
    //    cin >> filename;
    char* filename = argv[1];
    unsigned int frequencies[UniqueSymbols] = {0};
    run(filename,frequencies);
//    for(int i = 0; i < 256; i++)
//        cout << frequencies[i] << endl;
    //    cout << UniqueSymbols << endl;
    //    const char* ptr = SampleString;
    //    while (*ptr != '\0')
    //        ++frequencies[*ptr++];

    INode* root = BuildTree(frequencies);

    HuffCodeMap codes;
    GenerateCodes(root, HuffCode(), codes);
    delete root;
    int codewords[UniqueSymbols] = {0};
    int cwlens[UniqueSymbols] = {0};

    for (HuffCodeMap::const_iterator it = codes.begin(); it != codes.end(); ++it)
    {
        unsigned int count = distance(it->second.begin(),it->second.end());
        for(int i = 0; i < count; i++)
            if(it->second[i]) 
                codewords[(unsigned int)(it->first)]+=(unsigned int)pow(2.0f,(int)count - i - 1);
        cwlens[(unsigned int)(it->first)]=count;
    }
    for(int i = 0; i < 256; i++)
    {
        if(cwlens[i]) 
        {
            cout << i << " " << frequencies[i] << " "; 
//            printBits(codewords[i],cwlens[i]);
            cout << endl;
        }
    }
    return 0;
}
