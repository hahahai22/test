#include <iostream>
#include <string>
#include <utility>

class MyClass
{
public:
    std::string data;

    // 普通构造函数
    MyClass(const std::string &str) : data(str)
    {
        std::cout << "constructed: " << data << std::endl;
    }

    // 移动构造函数
    MyClass(MyClass&& other) noexcept : data(std::move(other.data))
    {
        std::cout << "moved: " << data << std::endl;
    }

    // // 拷贝构造函数
    // MyClass(const MyClass &other) : data(other.data)
    // {
    //     std::cout << "copy construction: " << data << std::endl;
    // }
};

int main()
{
    MyClass obj2("Hello World!");  // 调用普通构造函数
    // MyClass obj3 = std::move(obj2);  // 调用移动构造函数

    // std::cout << "obj2 data: " << obj2.data << std::endl;
    // std::cout << "obj3 data: " << obj3.data << std::endl;

    return 0;
}
