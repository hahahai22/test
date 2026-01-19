#include <iostream>

// 普通全局变量：外部链接，可被其他文件通过extern访问
int globalVar = 10;

// 静态全局变量：内部链接，只在本文件可见
static int staticGlobalVar = 20;

void printVars()
{
    std::cout << "From file1: globalVar = " << globalVar
              << ", staticGlobalVar = " << staticGlobalVar << std::endl;
}
