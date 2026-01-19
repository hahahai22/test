#include <iostream>

// 声明普通全局变量（来在file1.cpp文件），访问的是file1的变量
extern int globalVar; // 这里只是引用
void printVars();  // 这里只是告诉编译器函数存在，定义在别处

// 尝试声明静态全局变量（编译报错，因为在file2中对file1中的静态全局变量不可见）
// extern static int staticGlobalVar;

int main() { 
    std::cout << "From file2: globalVar = " << globalVar << std::endl;
    globalVar = 30;
    printVars();
    return 0;
}
