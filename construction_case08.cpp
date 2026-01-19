#include <iostream>
#include <string>
using namespace std;

class Point
{
public:
    Point(int xx, int yy) // default construction
    {
        this->x = xx;
        this->y = yy;
        cout << "Calling default construction!!!" << endl;
    }

    Point(Point &p)
    {
        this->x = p.x;
        this->y = p.y;
        cout << "Calling copy construction!!!" << endl;
    }

    int getX() { return x; }
    int getY() { return y; }

private:
    int x;
    int y;
};

void func01(Point p) // 将p3赋值给p对象----使用p3对象初始化p对象
{
    cout << p.getX() << endl;
}

Point func02()
{
    Point a(21, 2);
    return a;  // 局部对象赋值给全局对象
}

// 1. 使用p1对象初始化p2对象
void test01()
{
    Point p1(11, 1);
    Point p2(p1);
    /*
    Calling default construction!!!
    Calling copy construction!!!
    */
}

// 2. 对象p3作为函数的实参
void test02()
{
    Point p3(13, 3);
    func01(p3);
    /*
    Calling default construction!!!
    Calling copy construction!!!
    13
    */
}

// 3. 类对象作为返回值
void test03()
{
    Point b(17, 19);
    b = func02();
    cout << b.getX() << endl;
    /*
    Calling default construction!!!
    Calling default construction!!!
    21
    */
}

int main(int argc, char const *argv[])
{
    // test01();
    // test02();
    test03();
    return 0;
}
