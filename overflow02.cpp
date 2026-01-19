#include <stdio.h>
#include <float.h>
#include <math.h>

int main(int argc, char const *argv[])
{
    float large_float = FLT_MAX;
    float overflow_result = large_float * 2.0;

    printf("large_float: %e\n", large_float);
    printf("overflow result: %e\n", overflow_result);

    if (overflow_result == INFINITY)
    {
        printf("result is Infinity\n");
    }
    /*
    输出结果：

    large_float: 3.402823e+038
    overflow result: 1.#INF00e+000
    result is Infinity
    */

    return 0;
}
