#include <iostream>

class Base
{
public:
    virtual void foo() {} // 虚函数，引入vptr
    int x;                // 非静态成员变量
};

class Derived : Base
{
public:
    int y;
};

int main(int argc, char const *argv[])
{
    Base b;
    std::cout << sizeof(b) << std::endl; // 16  vptr占用8字节，x占用4字节；对齐填充4字节，共16字节。
    /*
        |      vptr (8字节)       |
        | x（4字节）| 填充（4字节）|
    */
}

/*
类的内存布局（成员函数，成员变量）：
    成员函数：
    所有成员函数（包括静态/非静态）的机器码存储在程序的代码区（.text段）
    * 非静态成员函数 隐式包含this指针参数，但函数本身不占用对象内存
    * 虚函数通过虚函数表（vtable）间接调用，对象需存储指向 vtable 的指针（vptr）。
    静态成员变量：存储在全局/静态区（.data或.bss段），与类的实例无关，所有对象共享同一份静态成员变量
    非静态成员变量：存储位置取决于对象存储的位置（栈，堆，全局区）

*/

