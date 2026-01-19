#include <iostream>

class Empty
{
};

int main(int argc, char const *argv[])
{
    Empty e1, e2;
    std::cout << &e1 << " " << &e2 << std::endl; // 地址相差1字节
    std::cout << sizeof(e1) << std::endl;        // 1
}

/*
C++标准规定，不同对象必须拥有自己唯一的内存地址。若空对象大小为0字节，则可能出现多个对象共享同一地址的情况，则与前面结论相悖。
编译器为每个空对象插入1字节的占位符，确保地址唯一性。
*/
