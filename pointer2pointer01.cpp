#include <iostream>

/*
指向C字符串（char*）数组的指针
*/
int main(int argc, char** argv)
{
    std::cout << "共传入 " << argc << " 个参数（包括程序名）\n";
    for (int i = 0; i < argc; ++i)
    {
        std::cout << "argv[" << i << "] = \"" << argv[i] << "\"\n";
        
        
    }
    return 0;
}
