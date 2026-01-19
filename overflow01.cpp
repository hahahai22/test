#include <stdio.h>

int main(int argc, char const *argv[])
{
    long long_value = 2147483648;
    int int_value = (int)long_value;
    printf("%d\n", int_value);
    return 0;
    /*
    输出：
    -2147483648
    */
}
