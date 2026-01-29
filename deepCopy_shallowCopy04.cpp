#include <iostream>
#include <cstring> // for strcpy, strlen


/// @brief 对03中 浅拷贝构造问题导致的double free问题进行修复
/// 增加深拷贝构造

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

    // 深拷贝构造：
    MyClass(MyClass& other)
    {
        data = new char[strlen(other.data) + 1];
        strcpy(data, other.data);
    }

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
    MyClass obj3("DeepCopy");
    // 这里调用深拷贝构造
    MyClass obj4 = obj3;
    obj4.print();

    return 0;
}
