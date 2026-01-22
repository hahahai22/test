#include <iostream>

class base
{
public:
    int b = 20;

private:
    int a = 10;
};

class derive : public base
{
public:
    int b = 30;
};

int main(int argc, char** argv)
{

    base* B = new derive();
    std::cout << B->b << std::endl;
};
