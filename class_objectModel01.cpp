class MyClass
{
public:
    int x; // 非静态成员变量，存储位置跟随对象
};

void foo()
{
    MyClass a;                  // 栈区。成员x在栈区
    MyClass *b = new MyClass(); // 堆区。成员x在堆区
    static MyClass c;           // 全局区/静态区。成员x在全局区/静态区
}
