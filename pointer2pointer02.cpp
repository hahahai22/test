#include <iostream>

void allocate(int** p)
{
    *p = (int *)malloc(sizeof(int));
    **p = 42;
}

/// 函数中修改指针的值
int main(int argc, char** argv)
{
    int* ptr = nullptr;
    allocate(&ptr);

    std::cout << *ptr << std::endl;

    free(ptr);
}
