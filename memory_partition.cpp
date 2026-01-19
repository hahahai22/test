// main.cpp
#include <iostream>

int global_init = 42;      // 存储在 .data 段
int global_uninit;         // 存储在 .bss 段

void foo(int x) {
    static int static_var = 10;  // 存储在 .data 段（已初始化静态变量）
    int local_var = x + 1;       // 存储在栈区
    std::cout << local_var << std::endl;
}

int main() {
    const char* msg = "Hello";   // 字符串常量存储在 .rodata 段
    foo(global_init);
    std::cout << msg << std::endl;
    return 0;
}


/*
1. 内存分区：
文本区

全局区

栈区
堆区
.data   段
.bss    段 最初是源于 汇编器的伪指令，声明未初始化的静态内存块（Block Started by Symbol）存储未初始化的全局变量和静态变量（包括显式初始化为0的变量）
.rodata 段  （read only data）

*/