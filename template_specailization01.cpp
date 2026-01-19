#include <iostream>
using namespace std;

/*
完全特化（full specailization）局部特化（patial specailization）
*/

// 默认版本：通用
template <typename DataType, typename Policy, int Size>
class StoragePolicy
{
public:
    void info()
    {
        cout << "Generic storage, type=" << typeid(DataType).name()
             << " Policy=" << typeid(Policy).name() << " Size=" << Size << endl;
    }
};

/*
tag type 类型标签，只是为了区分不同的（特化）实现，编译器可以根据这个类型不同，选择不同的特化实现（或者函数重载）。
*/
struct LDSNativeRead
{
};

template <typename DataType, int Size>
class StoragePolicy<DataType, LDSNativeRead, Size>
{
public:
    void info()
    {
        cout << "Specialized storage for LDSNativeRead, type=" << typeid(DataType).name()
             << " size=" << Size << endl;
    }
};

int main()
{
    StoragePolicy<float, double, 16> generic;
    generic.info();

    StoragePolicy<int, LDSNativeRead, 32> specialized;
    specialized.info();
}
