#include <iostream>
using namespace std;

/**
 * C++11 引入的统一初始化语法（使用大括号 {}）旨在提供一种一致、安全的初始化方式。
 * 它可以用于初始化标量、数组、容器、结构体/类，甚至嵌套结构。
 * 相比旧式的圆括号 () 或赋值 =，它更严格（避免窄化转换，如 double 到 int 的隐式截断），并支持
 * initializer_list 机制。
 */

/// @brief 初始化基本数据类型，标量
int main(int argc, char** argv)
{
    int x{10};
    double y{3.14};
    // int z{3.14}; // 编译报错，narrowing conversion
    cout << x << ", " << y << endl;
}
