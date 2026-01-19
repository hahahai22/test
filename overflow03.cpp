#include <stdio.h>
#include <limits.h>
#include <cmath>

int main(int argc, char const *argv[])
{
    int max_int = INT_MAX;
    int result = max_int + 2;
    printf("int max value: %d\n", max_int);
    printf("result: %d\n", result);
    /*
    输出结果：
    int max value: 2147483647
    result: -2147483647
    */
    return 0;
}
