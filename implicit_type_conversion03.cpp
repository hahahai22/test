#include <stdio.h>

int main()
{
    int a = -1;         // 有符号整数 0xFFFF FFFF
    unsigned int b = 1; // 无符号整数

    if (a > b)
    // if (a > (int)b)  // 解决方案
    {
        printf("a > b\n");
    }
    else
    {
        printf("a <= b\n");
    }
    /*
    输出结果：a > b
    分析：在a,b进行比较时，a发生了隐式类型转换，转换为unsigned integer
    -1的计算机表示采用补码（整数采用补码表示）：
    0x1000 0001 // 原码
    0xFFFF FFFE // 反码
    0xFFFF FFFF // 补码
    因此a整型 隐式转换之后的结果为：4,294,967,295
    */
    return 0;
}
