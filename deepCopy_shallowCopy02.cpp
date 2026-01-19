#include <iostream>
#include <cstring> // for strcpy, strlen

class MyClass {
private:
    char* data; // 动态分配的内存

public:
    // 构造函数
    // MyClass(const char* str) {
    //     data = new char[strlen(str) + 1]; // 动态分配内存
    //     strcpy(data, str);
    // }
    MyClass()
    {
        std::cout << "构造函数" << std::endl;
    }

    // 浅拷贝：默认的拷贝构造函数
    // MyClass(const MyClass& other) = default;

    // 深拷贝：自定义的拷贝构造函数
    // MyClass(const MyClass& other) {
    //     data = new char[strlen(other.data) + 1]; // 分配新的内存
    //     strcpy(data, other.data); // 复制数据
    // }

    // 浅拷贝：默认的赋值操作符
    // MyClass& operator=(const MyClass& other) = default;

    // 深拷贝：自定义的赋值操作符
    // MyClass& operator=(const MyClass& other) {
    //     if (this != &other) { // 防止自赋值
    //         delete[] data; // 释放旧内存
    //         data = new char[strlen(other.data) + 1]; // 分配新的内存
    //         strcpy(data, other.data); // 复制数据
    //     }
    //     return *this;
    // }

    // 打印数据
    void print() const {
        std::cout << "Data: " << data << std::endl;
    }

    // 析构函数
    // ~MyClass() {
    //     delete[] data; // 释放内存
    // }
};

int main() {

    MyClass arr();
    // // 浅拷贝测试
    // MyClass obj1("Hello");
    // MyClass obj2 = obj1; // 浅拷贝：指向相同的内存
    // obj2.print();

    // // 修改 obj2 的数据时，也会影响 obj1
    // // 因为它们指向相同的内存地址（在浅拷贝的情况下）
    // obj2 = MyClass("World");
    // obj1.print(); // obj1 也被修改（浅拷贝的效果）


    // // 深拷贝测试
    // MyClass obj3("DeepCopy");
    // MyClass obj4 = obj3; // 深拷贝：创建了独立的副本
    // obj4.print();

    // // 修改 obj4 的数据时，不会影响 obj3
    // obj4 = MyClass("NewDeepCopy");
    // obj3.print(); // obj3 没有被修改（深拷贝的效果）

    return 0;
}
