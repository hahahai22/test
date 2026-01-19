#include <stdio.h>

union Data
{
    int i;
    float f;
    char str[20];
};

int main()
{
    union Data data;
    data.i = 10;
    printf("data.i: %d\n", data.i); // data.i: 10

    data.f = 2.5;
    printf("data.f: %f\n", data.f); // data.f: 2.500000

    printf("data.i: %d\n", data.i); // data.i: 1075838976

    return 0;
}