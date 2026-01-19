#include <stdio.h>

int main(int argc, char const *argv[])
{
    char a = 1;
    printf("%u", sizeof(a));
    printf("\n");
    printf("%u", sizeof(+a));   // 单操作符+，一个主要作用就是“整型提升”
    /*
    输出：
    1
    4
    */
    return 0;
}

