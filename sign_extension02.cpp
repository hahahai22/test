#include <stdio.h>

#define GET_LSB(x) (x & 0x0000FFFF)
#define GET_MSB(x) ((x & 0xFFFF0000) >> 16) & 0x0000FFFF

/*
位操作中符号扩展问题：
*/
int main()
{
    unsigned int a = 0xF0000000; // 一个高位为1的数
    short b = GET_LSB(a);
    printf("GET_LSB(a) = %d\n", b);

    /*
    详细说明这部分：
    GET_MSB(a)宏的结果hexadecimal：0xF000 binary： 1111 0000 0000 0000
    将GET_MSB(a)赋值给short类型（16位有符号整数），在16位有符号整数中，最高位是符号位。
    在计算机中，负数是通过补码表示的（对一个数绝对值取反加1，表示负数）
    
    对结果-4096进行说明：
    1111 0000 0000 0000 隐式转换为 16位有符号整数 即，因为是有符号，因此按位取反再加一进行 binary表示：1001 0000 0000 0000 结果转为decimal -4096
    */
    short c = GET_MSB(a);   // 1111 0000 0000 0000
    printf("GET_MSB(a) = %d\n", c);
    /*
    输出结果：
    GET_LSB(a) = 0
    GET_MSB(a) = -4096
    */
}

