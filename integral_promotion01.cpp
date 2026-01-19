#include <stdio.h>

int main(int argc, char const *argv[])
{
    char a = 0xb6;
    short b = 0xb600;
    int c = 0xb6000000;

    if (a == 0xb6)
        printf("a");
    if (b == 0xb600)
        printf("b");
    if (c == 0xb6000000)
        printf("c");

    /*
    最后输出结果为c
    因为整型提升：a/b整型提升为int类型
    */
    return 0;
}
