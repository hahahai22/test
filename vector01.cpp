#include <vector>
#include <iostream>

void printVector(std::vector<int> &v)
{
    for (std::vector<int>::iterator it = v.begin(); it != v.end(); it++)
    {
        std::cout << *it << std::endl;
    }
    
}

void test01()
{
    std::vector<int> v1;
    for (int i = 0; i < 10; i++)
    {
        v1.push_back(i);
    }
    printVector(v1);

    std::vector<int> v2(4, 100);
    printVector(v2);

    std::vector<int> v3(v1.begin(), v1.end());
    printVector(v3);
}


int main()
{
    test01();
    return 0;
}
