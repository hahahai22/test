#include <iostream>

class FOO
{
public:
    int normalMemberVar;        // 普通成员变量，每个对象一份
    static int staticMemberVar; // static成员变量，属于类本身，所有对象共享一份。

    void normalMemberFunc() // 普通成员函数：需实例调用
    {
        std::cout << "normalMemberVar: " << normalMemberVar << std::endl;
    }

    static void staticMemberFunc() // static 成员函数：通过类名调用/也可以通过实例调用
    {
        std::cout << "staticMemberVar: " << staticMemberVar << std::endl;
    }
};

// 静态成员变量需要：类内声明，类外初始化
int FOO::staticMemberVar = 100;

int main(int argc, char** argv)
{
    FOO obj1, obj2;
    obj1.normalMemberVar = 1;
    obj2.normalMemberVar = 2;
    /// static变量无需实例即可访问
    FOO::staticMemberVar = 200; // 修改共享变量

    obj1.normalMemberFunc();
    obj2.normalMemberFunc();

    FOO::staticMemberFunc(); // 输出: Static member func, staticVar: 200（无需对象）
    obj1.staticMemberFunc(); // 也可通过对象调用，但本质是类的
    return 0;
}
