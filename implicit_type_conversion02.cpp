#include <iostream>

int main(int argc, char const *argv[])
{
    unsigned int a = 10000U;
    // int b = -1;
    int b = static_cast<unsigned int>(-1); //使用显式类型转换避免隐式类型转换存在的问题

    if (a > b)
    {
        std::cout << "a > b" << std::endl;
    }
    else
    {
        std::cout << "a < b" << std::endl;
    }
    
    return 0;
}
