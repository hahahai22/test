#include <iostream>

// 递归说明
unsigned long long fractorial(int n)
{
    if (n == 0)
    {
        return 1; // 基本情况：当n为0时，返回1
    }
    else
    {
        return n * fractorial(n - 1);
    }
}

int main(int argc, char const *argv[])
{
    int num = 5;
    std::cout << "fractorial of " << num << " is " << fractorial(num) << std::endl;

    return 0;
}
