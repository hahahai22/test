#include <stdio.h>

#define GET_LSB(x) (x & 0x0000FFFFU)
#define GET_MSB(x) ((x & 0xFFFF0000U) >> 16) & 0x0000FFFFU
using uint = unsigned int;
/*
位操作中符号扩展问题：
*/

int main()
{
    unsigned int a = 0xF0000000; // 一个高位为1的数
    unsigned short b = GET_LSB(a);
    unsigned short c = GET_MSB(a);
    printf("GET_LSB(a) = %u\n", b);
    printf("GET_MSB(a) = %u\n", c);
    return 0;
    /*
    输出结果：
    GET_LSB(a) = 0
    GET_MSB(a) = 61440 
    */
}
