#include <stdio.h>
#include <iostream>

int main(int argc, char const *argv[])
{
    int a = -5;
    unsigned int b = 10;
    int result = a + b;

    std::cout << result << std::endl;

    printf("result: %d\n", result);
    /*
    这里为什么输出结果是正确的，原因在于：printf的%d格式符，结果被解释为有符号整数，
    即(1111 1111 1111 1111 1111 1111 1111 1011)_2 + (10)_10 解释为有符号整数，这个二进制超出unsigned integer表示范围，解释为负数。
    */
    return 0;
}
