#include <iostream>

namespace colorspace {
enum Color
{
    Red,
    Blue,
    Green
};
} // namespace colorspace

int main(int argc, char** argv)
{
    using namespace colorspace;
    Color c = Red;
    if(c == Blue)
    {
        std::cout << "blue" << std::endl;
    }
    else if(c == Red)
    {
        std::cout << "red" << std::endl;
    }
    return 0;
}
