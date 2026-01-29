#include <iostream>
#include <cstring> // for strcpy, strlen

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
    // 这里调用拷贝构造，因为没有深拷贝构造，编译器自动生成浅拷贝构造
    // obj3, obj4指向同一块内存，
    // obj4析构-》delete data[]
    // obj3析构-》delete data[] -》double free  -》trace trap崩溃
    MyClass obj4 = obj3;
    obj4.print();

    return 0;
}
