#include <iostream>
#include <vector>
using namespace std;

/// @brief 初始化std::vector容器
int main(int argc, char** argv)
{
    std::vector<int> vec{1, 2, 3, 4};
    for(int& num : vec)
    {
        cout << num << " ";
    }
    cout << endl;
    return 0;
}
