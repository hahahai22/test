#include <stdio.h>

// #define GET_LSB(x) (x & 0x0000FFFF) // 没有U后缀
#define GET_LSB(x) (x & 0x0000FFFFU)
#define GET_MSB(x) ((x & 0xFFFF0000U) >> 16)
using uint = unsigned int;

int main(int argc, char const *argv[])
{
    int x = -1; // 0xFFFF FFFF
    int lsb = GET_LSB(x);
    printf("x = %d, LSB = %d\n", x, lsb);

    int msb = GET_MSB(x);
    printf("x = %d, MSB = %d\n", x, msb);
    return 0;
    /*
    输出结果：
    x = -1, LSB = 65535
    x = -1, MSB = 65535
    */
}

