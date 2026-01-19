#include <iostream>
#include <memory>

using namespace std;
class B;

/*
class A
{
public:
    shared_ptr<B> b_ptr;
    ~A() { cout << "A 析构" << endl; }
};

class B
{
public:
    shared_ptr<A> a_ptr;
    ~B() { cout << "B 析构" << endl; }
};

int main(int argc, char** argv)
{
    shared_ptr<A> a = make_shared<A>(); // ref_A = 1
    shared_ptr<B> b = make_shared<B>(); // ref_B = 1

    a->b_ptr = b; // ref_B = 2
    b->a_ptr = a; // ref_A = 2

    return 0;
}
*/

class A
{
public:
    shared_ptr<B> b_ptr;
    ~A() { cout << "A 析构" << endl; }
};

class B
{
public:
    weak_ptr<A> a_ptr;
    ~B() { cout << "B 析构" << endl; }
};

int main(int argc, char** argv)
{
    shared_ptr<A> a = make_shared<A>(); // ref_A = 1
    shared_ptr<B> b = make_shared<B>(); // ref_B = 1

    a->b_ptr = b; // ref_B = 2
    b->a_ptr = a; // ref_A = 1

    return 0;

    /*
    退出作用域：
    局部变量a,b销毁，A，B引用计数减1。
    B对象持有A对象的a_ptr因为是weak_ptr所以引用计数为1，减1之后，B对象析构。
    A对象持有B对象的b_ptr，因为B析构，导致A对象里面的b_ptr也被销毁。所以，A的引用计数从1减为0。

    从始至终只有A，B两个对象a, b；A，B对象的引用计数的变化。
    
    */
}
