#include <iostream>
using namespace std;

/// @brief 初始化数组
int main(int argc, char** argv)
{
    int arr1[3]{1, 2, 3};
    int arr2[]{4, 5, 6, 7}; // // 自动推导大小为 4
    int arr3[5]{8, 9};

    for(int i : arr3)
    {
        cout << i << ", ";
    }
}
