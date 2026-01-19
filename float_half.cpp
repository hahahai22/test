#include <stdio.h>

int main(int argc, char const *argv[])
{   
    short arr[2] = {3, 2};
    short *c = arr;

    float *a = (float*)c;

    short *b = (short *)&a;
    printf("b: %d, %d\n", *b, b+1);
    return 0;
}
