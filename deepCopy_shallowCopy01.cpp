#include <iostream>
#include <cstring>

class MyClass
{
public:

    char *data;

    MyClass(const char* str)
    {
        data = new char[strlen(str) + 1];
        strcpy(data, str);
    }

    

};
