#include <iostream>

void foo()
{
    int localVar              = 0; // 普通局部变量，函数每次调用都初始化为0
    static int staticLocalVar = 0; // static局部变量，只初始化一次为0

    localVar++;
    staticLocalVar++;

    std::cout << "localVar = " << localVar << "; staticLocalVar = " << staticLocalVar << std::endl;
}

int main(int argc, char** argv)
{
    foo(); // 输出 ： localVar = 1; staticLocalVar = 1
    foo(); // 输出 ： localVar = 1; staticLocalVar = 2
    foo(); // 输出 ： localVar = 1; staticLocalVar = 3
    return 0;
}
