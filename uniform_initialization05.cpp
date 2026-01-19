#include <iostream>
#include <string>
using namespace std;

/// @brief 初始化结构体或着类

struct Point
{
    int x;
    double y;
    std::string lable;
};

int main(int argc, char** argv)
{
    Point p{10, 20.5, "orign"};
    cout << "x: " << p.x << "; y: " << p.y << "; lable: " << p.lable << endl;
    return 0;
}
