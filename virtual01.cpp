#include <iostream>
using namespace std;

// 静态多态 - 重载
class Math
{
public:
    static int multiply(int a, int b) { return a * b; }
    static double multiply(double a, double b) { return a * b; }
};

// 动态多态 - 重写
class Animal
{
public:
    virtual void speak() { cout << "Animal speak" << endl; }
    virtual ~Animal() = default;
};

class Cat : public Animal
{
public:
    void speak() override { cout << "Meow!" << endl; } // C++11使用override
};

class Dog : public Animal
{
public:
    void speak() override { cout << "Woof!" << endl; }
};

int main(int argc, char **argv)
{
    // 静态多态使用
    cout << Math::multiply(2, 3) << endl;
    cout << Math::multiply(2.5, 3.5) << endl;

    // 动态多态使用
    Animal *animal1 = new Cat();
    Animal *animal2 = new Dog();
    animal1->speak();
    animal2->speak();
    return 0;
}
/*
6
8.75
Meow!
Woof!
*/