#include <iostream>

struct struct_size01
{
    void *input_addr;
    void *weight_addr;
    void *output_addr;
    void *workspace_addr;
    int n;
    int c;
    int h;
};

int main(int argc, char const *argv[])
{
    std:: cout << sizeof(struct_size01) << std::endl;  // 48===>因为内存对齐
    /*
    _ _ _ _  _ _ _ _
    _ _ _ _  _ _ _ _
    _ _ _ _  _ _ _ _
    _ _ _ _  _ _ _ _
    _ _ _ _  _ _ _ _
    _ _ _ _ “_ _ _ _”  // 这4字节为内存对齐所填充
    */
    return 0;
}

