#include <iostream>

// 普通函数：外部链接，可被其他文件调用
void normalFunc() { std::cout << "Normal function from file1" << std::endl; }

// static 函数：内部链接，只在本文件可见
static void staticFunc() { std::cout << "Static function from file1" << std::endl; }

void callFuncs()
{
    normalFunc();
    staticFunc();
}
