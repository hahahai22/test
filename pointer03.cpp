#include <iostream>

union FP16
{
    unsigned short u;
    struct
    {
        unsigned int Mantissa : 10;
        unsigned int Exponent : 5;
        unsigned int Sign : 1;
    };
};

int main(int argc, char const *argv[])
{
    /*
    16.032226的二进制为: 0100,0001,1000,0000,0100,0010,0000,0000
    short 类型为一位sign Bit, 15位 Value Bits。
    存入内存的二进制没有变化，只是对二进制的解释不一样。
    打印低16位 LSB
    */
    float a = 16.032226;          // 0x41804200
    short low = *(short *)(&(a)); // 0100,0010,0000,0000
    printf("low: %d\n", low);     // 16896

    /*
    打印高16位 MSB
    */
    short high = *((short *)(&a) + 1); // 0100,0001,1000,0000
    printf("high: %d\n", high);        // 16768

    return 0;
}
