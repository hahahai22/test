#include <stdio.h>

int main()
{
    int x = 10;
    int *p = &x;

    printf("x 的值： %d", x);
    printf("x 的首地址（地址）： %p\n", &x);
    printf("p 的值（x的地址/首地址）： %p\n", p);
    printf("*p 的值： %d\n", *p);
    
    return 0;
}
