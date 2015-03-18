#include <iostream>
#include "print_helpers.h"
#include "uint256.h"
//#include "testdatagen.h"
#include <limits.h>
#include <cmath>

//template <typename T>
//inline unsigned int castToUint(T val)
//{
//  unsigned int result = 0;
//  for (int i = 0; i < 32; i++)
//    {
//      if ((val & 1) == 1) result += (unsigned int)pow(2,i);
//      val.operator>>=(1);
//    }
//  return result;
//}
//

using namespace std;

int main(void)
{
  unsigned int i = 1;
//  print32Bits(i%UINT_MAX); 
//  i >>= 1;
//  unsigned int b = *reinterpret_cast<unsigned int*>(&i);
  print32Bits(i<<35);
//  print32Bits((unsigned int)i);
}
