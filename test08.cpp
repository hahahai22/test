#include <iostream>
#define PADDING_VALUE 8

int main(int argc, char** argv)
{
    int align_padding = (3 + PADDING_VALUE - 1) & (~(PADDING_VALUE - 1));
    std::cout << align_padding << std::endl;
}
