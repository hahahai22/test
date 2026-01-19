#include <iostream>
#include <vector>
#include <string>
using namespace std;

/// @brief 初始化嵌套结构（二维容器）
int main(int argc, char** argv)
{
    std::vector<std::vector<int>> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    for(std::vector<int> row : matrix)
    {
        for(int num : row)
        {
            cout << num << " ";
        }
        cout << endl;
    }
    return 0;
}
