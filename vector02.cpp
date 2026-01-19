
#include <iostream>
#include <vector>

int main(int argc, char const* argv[])
{
    std::vector<int> v = {2, 4, 8, 30};
    std::cout << v.size() << "\n"
              << v.data() << "\n"
              << v.front() << "\n"
              << v[0] << "\n"
              << v.back() << "\n"
              << *v.begin() << "\n"
              << *(v.end() - 1) << std::endl;

}
/*
4
0x1f5ffbf2f70
2
2
*/
