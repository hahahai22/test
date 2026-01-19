#include <iostream>
#include <utility>
using namespace std;
#include <string>

class Person
{
public:
    string name;
    int age;

    Person(Person &&p) noexcept : name(std::move(p.name)), age(std::move(p.age))
    {
        this->name = p.name;
        this->age = p.age;
    }

    Person(string name, int age)
    {
        this->name = name;
        this->age = age;
    }

    Person(){}
};

int main(int argc, char const *argv[])
{
    Person p1("zhangsan", 18);
    cout << "1: " << p1.age << endl;
    Person p2(std::move(p1));

    cout << p1.age << endl;

    return 0;
}

