#include <iostream>

int main(int argc, char const *argv[])
{
    unsigned int a = 10000U;
    int b = -1;
    /*
    详细说明：-1在计算机中以二进制补码的方式存储，即 1111 1111 1111 1111 1111 1111 1111 1111。
    int b = -1;        将 1111 1111 1111 1111 1111 1111 1111 1111 按照int类型解释，即-1
    转成unsigned int即，将 1111 1111 1111 1111 1111 1111 1111 1111 按照unsigned int类型解释，即4,294,967,295
    */
    if (a > b) // 这里进行比较的时候，b隐式的转成了a的类型unsigned int类型。即：b 变成 2^32大小
    {
        std::cout << "a > b" << std::endl;
    }
    else
    {
        std::cout << "a < b" << std::endl;
    }
    
    return 0;
}
