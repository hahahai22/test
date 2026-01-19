#include <iostream>

// 声明普通函数（来自 file1），可调用
extern void normalFunc();
extern void callFuncs();

// 尝试声明 static 函数（编译错误，因为不可见）
// extern void staticFunc();  // 错误：未定义的标识符

int main()
{
    normalFunc(); // 输出: Normal function from file1
    // staticFunc();  // 编译错误
    callFuncs(); // 通过 file1 的函数间接调用（但 staticFunc 只在 file1 内可见，这里正常Func
                 // 可调用）
    return 0;
}
