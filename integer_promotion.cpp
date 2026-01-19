#include <iostream>

/*
 * 整型提升（Integer Promotion）：
 */
// 这里加上U的原因是因为，防止x如果为sign int，编译器进行的将常量0x0000FFFF
#define GET_LSB(x) (x & 0x0000FFFFU)
#define GET_MSB(x) ((x & 0xFFFF0000U) >> 16) & 0x0000FFFFU
using uint = unsigned int;

int main()
{
    uint a = 262148;
    short b = GET_LSB(a);
    short c = GET_MSB(a);
    printf("%d\n", b);
    printf("%d\n", c);
}
