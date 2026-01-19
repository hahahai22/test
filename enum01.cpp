#include <iostream>
#include <stdio.h>

typedef enum
{
    MIOPEN_SOFTMAX_MODE_INSTANCE = 0,
    MIOPEN_SOFTMAX_MODE_CHANNEL  = 2,
} softMaxmode_t;

int main(int argc, char const* argv[])
{
    softMaxmode_t mode0 = static_cast<softMaxmode_t>(4);
    softMaxmode_t mode1 = MIOPEN_SOFTMAX_MODE_CHANNEL;

    std::cout << mode0 << std::endl;
    std::cout << mode1 << std::endl;
    // printf("%u\n", mode);
    return 0;
}
