#include <iostream>


class Add
{
public:
    int operator()(int a, int b)
    {
        return a + b;
    }
};

int main(int argc, char **argv)
{
    using namespace std;
    Add add;
    int ret = add(10, 11);

    cout << ret << endl;
    return 0;
}