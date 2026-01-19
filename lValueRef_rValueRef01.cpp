#include <iostream>
#include <utility>
using namespace std;

void processLValueRef(int &x)
{
    cout << "processing left value: " << x << endl;
    x += 5;
}

void processRValueRef(int &&x)
{
    cout << "processing right value: " << x << endl;
    x += 10;
}

int main(int argc, char const *argv[])
{
    int a = 10;             // a是左值
    int &lRef = a;          // lRef是a的左值引用
    processLValueRef(lRef); // 左值引用传递 lRef是左值引用
    processLValueRef(a);    // 左值传递，a是左值，匹配左值引用参数

    processRValueRef(20); // 20是右值，直接传递给右值引用参数

    int &&rRef = 30;                   // rRef是右值引用，但它本身是一个左值
    processRValueRef(std::move(rRef)); // 使用std::move将一个左值转换为右值

    return 0;
}
