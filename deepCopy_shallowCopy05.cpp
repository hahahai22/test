#include <iostream>
#include <cstring> // for strcpy, strlen

/// @brief 调用重载的赋值运算符
/// operate=
/// 对象已经存在，再赋值会调用赋值运算符
/// 对象的初始化调用拷贝构造

class MyClass
{
private:
    char* data; // 动态分配的内存

public:
    // 构造函数
    MyClass(const char* str)
    {
        data = new char[strlen(str) + 1]; // 动态分配内存
        strcpy(data, str);
    }

    MyClass()

    // 打印数据
    void print() const { std::cout << "Data: " << data << std::endl; }

    // 析构函数
    ~MyClass()
    {
        delete[] data; // 释放内存
    }
};

int main()
{
    MyClass obj1("DeepCopy");
    MyClass obj2 = {"init"};

    obj2 = obj1;

    return 0;
}
