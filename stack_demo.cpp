#include <iostream>

void func()
{
    asm("nop"); // 断点位置1：func入口
    volatile int a = 10;
    int c = a;
    asm("nop"); // 断点位置2：func内部
}

int main()
{
    asm("nop"); // 断点位置3：main入口
    volatile int x = 1, y = 2, z = 3;
    std::cout << x;
    asm("nop"); // 断点位置4：调用func前
    func();
    asm("nop"); // 断点位置5：func返回后
    return 0;
}
