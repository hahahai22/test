#include <iostream>

void foo(int *arr, int len)
{
    for (int i = 0; i < len; i++)
    {
        std::cout << arr[i] << std::endl;
    }
}

int main(int argc, char const *argv[])
{
    int arr[] = {0, 1, 2, 3, 4, 5, 6};
    int len = sizeof(arr) / sizeof(arr[0]);
    foo(arr, len);
}
