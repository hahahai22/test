#include <iostream>

int main(int argc, char** argv)
{
    // char* fruits[] = {"apple", "banana", "cherry"};
    // char** list = fruits;  // list为指向char指针的指针。

    // for (int i = 0; i < 3; ++i)
    // {
    //     printf("%s\n", list[i]);
    // }

    char* fruits[] = {"apple", "banana", "cherry"};
    char** list = fruits;  // list为指向char指针的指针。

    for (int i = 0; i < 3; ++i)
    {
        printf("%s\n", list[i]);
        std::cout << list[i] << std::endl;
    }

    return 0;
}

/*
apple
banana
cherry


1. char* fruits[] = {"apple", "banana", "cherry"}; 这是char类型的指针数组，则这个数组大小为3，里面存的是char指针。
三个指针分别指向"apple", "banana", "cherry"的首字符
2. list[0]为什么能输出"apple"，list[0]指向apple的首字符地址，但输出的是apple，因为printf("%s")利用这个地址为起点，自动在内存中向前遍历，逐个输出字符，直到遇到\0
为止。

list → [ptr0,       ptr1,        ptr2]    // char** 指向指针数组
        ↓           ↓            ↓
       "apple\0"   "banana\0"   "cherry\0"
        ↑           ↑            ↑
       char*       char*        char*
*/
