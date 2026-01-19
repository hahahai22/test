#include <iostream>

/*
对象的存储位置
*/
class MyClass
{
public:
    int a;
    static int b;
};

int MyClass::b = 0;

int main(int argc, char const *argv[])
{
    MyClass obj02;                           // 对象声明为局部变量（局部变量存储在栈区）
    std::cout << sizeof(obj02) << std::endl; // 4

    static MyClass obj01;                    //  对象声明为静态变量（静态变量存储在全局区/静态区）
    std::cout << sizeof(obj01) << std::endl; // 4

    MyClass *obj03 = new MyClass();          // 对象由new/malloc创建（存储在堆区）
    std::cout << sizeof(obj03) << std::endl; // ojb03是指针 8字节
}
/*
ojb01存储在全局区 b也存储在全局区，为什么不能影响obj01的大小。
答：b属于类本身，不占用对象ojb01的空间

C++对象模型规定，每个对象实例必须包含其所有非静态成员变量的独立副本。这些成员变量在内存中按声明顺序连续排列（受内存对齐影响）
静态成员变量属于类本身，所有对象共享同一份实例，因此不占用对象的内存空间
成员函数（包括静态/非静态）的代码是共享的，存储在代码区，不会影响对象大小
*/
