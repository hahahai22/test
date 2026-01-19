#include <iostream>
using namespace std;

/*
多态：
1. 动态多态，运行时多态
程序运行期确定要调用的函数（动态绑定）。通过虚函数（virtual func）和继承机制实现
2. 静态多态，编译时多态
函数重载和模板实现（静态绑定）

要使用基类的指针调用子类的方法

virtual的作用就是告诉编译器，这个函数将来可能被派生类重写，请启用动态绑定。

*/
class base
{
public:
    // virtual int foo() = 0;   // 纯虚函数（基类中没有实现）
    virtual void foo1() { cout << "hello" << endl; } // 虚函数
    void foo2() { cout << "ni " << endl; }
};

class derive : public base
{
public:
    // 虚函数/纯虚函数在派生类中有重写
    void foo1() override { cout << "world" << endl; }
    void foo2() { cout << "hao " << endl; }
};

int main(int argc, char const* argv[])
{
    base* a = new derive(); // 父类指针指向子类对象
    a->foo1();
    /*
    这里静态绑定父类的foo2()，不能调用子类的方法，如果将父类的foo2()声明为virtual,子类重写父类的foo2()方法，并且可通过父类的指针调用子类的方法（动态绑定）
    */
    a->foo2();

    delete a;
    return 0;
}
