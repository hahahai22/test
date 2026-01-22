#include <iostream>
#include <map>
#include <string>

int main(int argc, char** argv)
{
    // 声明方式
    std::map<int, std::string> m0{{1, "one"}, {2, "two"}, {3, "three"}, {4, "four"}};

    std::map<int, std::string> m1;

    /// 插入方式：
    // 插入方式1 使用insert() 不可覆盖
    m1.insert({5, "five"});              
    m1.insert(std::make_pair(6, "six"));

    // 插入方式2 使用[]插入 覆盖
    m1[7] = "seven";                     

    // 遍历方式1
    /// 迭代器遍历
    for(auto it = m0.begin(); it != m0.end(); ++it)
    {
        std::cout << it->first << "->" << it->second << std::endl;
    }
    std::cout << "===================================" << std::endl;

    // 遍历方式2
    /// for范围
    for(const auto& kv : m0)
    {
        std::cout << kv.first << "->" << kv.second << std::endl;
    }
    std::cout << "===================================" << std::endl;

    /// operator[]访问
    std::cout << m0[3] << std::endl;

    /// 删除
    m0.erase(3);


}
