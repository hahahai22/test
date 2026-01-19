#include <iostream>
#include <cstring>
#include <cstdlib>

// ----------- 全局区 -----------
int global_var = 42;              // .data 区（已初始化的全局变量）
int global_bss;                   // .bss 区（未初始化的全局变量）
const char ro_str[] = "ReadOnly"; // .rodata 区（只读数据）
static int static_var = 99;       // .data 区（静态已初始化变量）

// ----------- 函数声明 -----------
void print_message(const char *msg);
int add_numbers(int a, int b);

int main()
{
    int local_var = 10;          // 栈区
    static int local_static = 5; // .data 区
    char local_array[20];        // 栈区
    strcpy(local_array, "StackString");

    int *heap_var = (int *)malloc(sizeof(int)); // 堆区
    *heap_var = 2025;

    print_message("Hello World");
    std::cout << "Sum = " << add_numbers(local_var, *heap_var) << std::endl;

    free(heap_var);
    return 0;
}

void print_message(const char *msg)
{
    std::cout << "Message: " << msg << std::endl;
}

int add_numbers(int a, int b)
{
    int sum = a + b; // 栈区
    return sum;
}
