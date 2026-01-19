#include <iostream>

class MyClass
{
public:
    int a;
    double b;
    static int c;
};

int main(int argc, char const *argv[])
{
    MyClass obj;
    std::cout << sizeof(obj) << std::endl;  // 16
    /*
    8字节对齐
    | a 4字节 | 填充 4字节  |
    |       b 8字节        |       
    */
}
