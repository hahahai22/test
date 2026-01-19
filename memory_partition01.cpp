#include "malloc.h"
#include <stdio.h>

int B_data;                  // .bss
static int B_data2;          // .bss
int D_val = 10;              // .data
extern const int R_val = 20; // .rodata
// constexpr int R_val = 20; // .rodata  如未进行使用，编译器会进行优化，nm不会显示
static int D_val2 = 30;      // .data

void func()
{
    int b_val;                   // stack
    int stack_val = 40;          // stack
    static int d_val = 50;       // .data
    static int b_val1;           // .bss
    const int r_data = 1;        // .rodata
    
    int *heap_val = new int(60); // deap  指针本身stack。指向的内存deap

    float *heap_val1 = (float *)malloc(sizeof(float)); // deap
    *heap_val1 = 3.14f;

    printf("&b_val   (stack)   = %p\n", (void*)&b_val);
    printf("&b_val1  (.bss)    = %p\n", (void*)&b_val1);
    printf("&d_val   (.data)   = %p\n", (void*)&d_val);
    printf("&r_data  (.rodata) = %p\n", (void*)&r_data);
    printf("&stack_val (stack) = %p\n", (void*)&stack_val);
    printf("heap_val (heap)    = %p\n", (void*)heap_val);
    printf("heap_val1 (heap)   = %p\n", (void*)heap_val1);

    delete heap_val1;
    free(heap_val);
}

/*
细致划分：
B/b BSS段           (elf节：.bss)(未初始化全局变量/静态变量)
D/d 数据段          (elf节：.data)(已初始化全局/静态变量) D(全局变量)：全局可见
d(静态变量)：仅文件可见 R/r 只读数据段       (elf节：.rodata) 常量 T/t
文本段/代码段    (elf节：.text) 栈区(stack) 未初始化/已初始化局部变量 堆区(heap)

其中：
文本区：.text
全局区：.data/.bss/.rodata
栈区：
堆区：
*/
