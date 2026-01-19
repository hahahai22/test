#include <stdio.h>
#include <stddef.h>

struct Example
{
    char a;  // 1
    int b;   // 4
    short c; // 2
};

int main(int argc, char const *argv[])
{
    struct Example e;
    printf("Size of struct Example: %d\n", sizeof(e));
    printf("Offset of a: %zu\n", offsetof(struct Example, a));
    printf("Offset of b: %zu\n", offsetof(struct Example, b));
    printf("Offset of c: %zu\n", offsetof(struct Example, c));

    return 0;
}
