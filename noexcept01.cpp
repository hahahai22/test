#include <iostream>
#include <vector>
#include <stdexcept>
#include <limits>

class MyTypeUnsafe
{
public:
    MyTypeUnsafe() = default;

    MyTypeUnsafe(MyTypeUnsafe &&) noexcept
    {
        std::cout << "move construction\n";
    }

    MyTypeUnsafe(MyTypeUnsafe &)
    {
        std::cout << "copy construction\n";
    }
};

int main(int argc, char const *argv[])
{
    std::vector<MyTypeUnsafe> vec;
    vec.reserve(1);
    vec.push_back(MyTypeUnsafe{});
}
// move construction
