#include <iostream>

int main(int argc, char const *argv[])
{
    /*
    i的内存首地址：0x61fe14，值是42
    i数据的内存中存储为：00000000 00000000 00000000 00100110
    */
    int i = 42;

    /*
    1. 取地址：integer_p存储的地址 0x61fe14；
    2. 取数据：根据integer_p的数据类型，从0x61fe14开始的4个字节（32位）数据
    3. 解释数据：根据integer_p的数据类型，将这32位数据按照int类型解释，得到一个整数
    */
    int *integer_p = &i;                  // 取地址
    std::cout << *integer_p << std::endl; // 取数据，解释数据
    // 42

    /*
    1. 取地址：fp存储的地址 0x61fe14；
    2. 取数据：根据fp的数据类型，从0x61fe14开始的4个字节（32位）数据
    3. 解释数据：根据fp的数据类型，将这32位数据按照float类型解释，得到一个浮点数
    0 00000000 00000000000000000100110
    解释为1位sign，8位exponent，23位mantissa
    */
    float *fp = (float *)&i;       // 取地址
    std::cout << *fp << std::endl; // 取数据，解释数据
    // 5.88545e-44

    /*
    因为指针，存储的是地址，因此这里i被解释为地址
    这里将i解释为“内存地址”大小是42，将其转成float类型的指针，赋值给float类型的指针fp
    内存地址是无效的地址，因为操作系统保留了0~255的内存地址
    */
    float *fp = (float *)i;
    return 0;
}
