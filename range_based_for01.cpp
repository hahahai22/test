#include <iostream>
#include <vector>

// 基于范围的for循环

int main(int argc, char const *argv[])
{
    /*
    for循环内，依次将vec的内容赋值给变量i，然后执行循环体内的语句。
    */
    int vec[] = {4, 9, 12, 7};
    for (auto i : vec)
    {
        i = 11; // 不能修改原容器中得值
        std::cout << i << "\n";
    }
    std::cout << "修改后：" // 上面for循环未修改原容器中的值
              << vec[0]
              << std::endl;

    std::vector<float> vec02 = {44, 78, 99, 21};
    for (auto &vec : vec02)
    {
        std::cout << vec
                  << std::endl;
    }
    

    return 0;
}
