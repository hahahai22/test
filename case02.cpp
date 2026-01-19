// #include <stdio.h>
// #include <limits.h>

// int main() {
//     int a = INT_MAX;
//     int b = a + 1;  // UB！可能输出任意值或崩溃
//     printf("%d\n", a);
//     printf("%d\n", b);
//     return 0;
// }

// #include <stdio.h>
// #include <string.h>

// void vulnerable() {
//     char buffer[4];
//     strcpy(buffer, "ABCDE"); // 写入 6 字节（包含 '\0'）
//     printf("%s\n", buffer);
// }

// int main() {
//     vulnerable();
//     return 0;
// }
// #include <stdio.h>

// int main(int argc, char const *argv[])
// {
//     int a = 10;
//     double b = 20.0;
//     a += b;
//     printf("sizeof(c): %d\n", sizeof(a));
//     return 0;
// }

#include <stdio.h>
#include <iostream>
using namespace std;

int main(int argc, char const *argv[])
{
    int a = -1;
    unsigned int b = 1;

    unsigned long long c = a * b;
    printf("%d\n", c);
    std::cout << c << std::endl;

    return 0;
}
