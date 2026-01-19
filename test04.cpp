#include <iostream>
using namespace std;

void swap_value(int a, int b);
void swap_address(int* a, int* b);

int main()
{
    int a = 10;
    int b = 20;
    //
    // swap_value(a, b);
    // cout << "a: " << a << " b: " << b << endl;

    //
    swap_address(&a, &b);
    cout << "a: " << a << " b: " << b << endl;

    return 0;
}

/*
按值传递：拷贝实参，修饰的是拷贝的实参
将main中实参的值拷贝一份，传递给swap_value的形参
*/
void swap_value(int a, int b)
{
    int tmp = a;
    a       = b;
    b       = tmp;
    cout << "a: " << a << " b: " << b << endl;
}

// void swap_address(int *a, int *b)
// {
//     int *tmp = a;
//     a = b;
//     b = tmp;
//     cout << "a: " << *a << " b: " << *b << endl;
// }

/*
按地址传递：修饰实参
操作的是内存中的实参a，b的数据
*/
void swap_address(int* a, int* b)
{
    int tmp = *a;
    *a      = *b;
    *b      = tmp;
    cout << "a: " << *a << " b: " << *b << endl;
}
