#include <iostream>
#include <utility>
#include <string>

class MyClass
{
public:
    std::string data;
    MyClass(const std::string &str) : data(str) // default构造函数，接受左值引用
    {
        std::cout << "constructed: " << data << std::endl;
    }

    // std::string &&是右值引用，表示这个构造函数专门用于处理临时对象或右值。
    // 使用std::move(str)是为了将str转换为右值引用，从而运行移动语义地使用，而不是复制数据
    MyClass(std::string &&str) : data(std::move(str)) // 移动构造函数，接受右值引用并使用std::move
    {
        std::cout << "moved: " << data << std::endl;
    }

    // // 拷贝构造函数应该接受 常量左值引用。这样是为了防止无意地修改传入的对象
    // MyClass(const MyClass &myclass)  // 接受 常量左值引用
    // {
    //     this->data = myclass.data;
    //     std::cout << "copy construction: " << this->data << std::endl;
    // }
};

int main()
{
    std::string myStr = "hello world!";
    MyClass obj1(myStr); // 传递的是左值 这里没有任何类型转换，只是将左值(myStr)绑定到左值引用(str)上。 调用默认构造函数

    MyClass obj2("Hello World!");   // 传递的右值 将右值("Hello World!")绑定到右值引用(str)上。调用移动构造函数
    MyClass obj3 = std::move(obj2); // obj2是左值，使用std::move转成右值。调用拷贝构造函数
    // MyClass obj4 = obj2;

    // std::cout << "obj1 data: " << obj1.data << std::endl;
    // std::cout << "obj2 data: " << obj2.data << std::endl;
    return 0;
}
