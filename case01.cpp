#include <iostream>
#include <stdio.h>
#include <atomic>

int main(int argc, char const *argv[])
{
    unsigned int a = 0x00020002;
    printf("a: %d\n", a);

    // float c = (float)(a & 0x0000ffff);
    // printf("c: %f\n", c);

    short d = (short)(a & 0xffff0000);
    printf("d: %f\n", d);

    // unsigned int b = 0x0002;
    // printf("b: %d\n", b);
    // atomic();


    return 0;
}

