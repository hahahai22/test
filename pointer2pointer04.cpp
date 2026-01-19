#include <iostream>

int main(int argc, char** argv)
{
    int a   = 10;
    int* p  = &a; // 定义一个指针 p，它指向 a 的地址
    int** q = &p; // 定义一个二级指针 q，它指向 p 的地址

    std::cout << a << std::endl;
    std::cout << *p << std::endl;
    std::cout << **q << std::endl; // 输出 q 所指向的指针所指向的变量的值，为 10

    std::cout << "Address of a: " << &a << std::endl;
    std::cout << "p: " << p << std::endl; // p 的值是 a 的地址
    std::cout << "Address of p: " << &p << std::endl;
    std::cout << "q: " << q << std::endl; // q 的值是 p 的地址
}
