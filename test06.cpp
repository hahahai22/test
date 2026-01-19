#include <iostream>
#include <stdio.h>
#define LDS_MAX_SIZE 15000
using namespace std;

int main(int argc, char const *argv[])
{
    // /bin/MIOpenDriver conv -n 1 -c 79 -H 181 -W 380 -k 192 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
    int n                   = 1;
    int c                   = 79;
    int h                   = 181;
    int w                   = 380;
    int k                   = 192;
    int r                   = 1;
    int s                   = 1;
    int p                   = 0;
    int q                   = 0;
    int stride_h            = 1;
    int stride_w            = 1;
    int dilate_h            = 1;
    int dilate_w            = 1;
    int group               = 1;
    int bacths_Single_block = 16;
    int group_single_block  = 1;

    int dilate_filter_h = dilate_h * (r - 1) + 1;
    int dilate_filter_w = dilate_w * (s - 1) + 1;

    int oh = (h + 2 * p - dilate_filter_h) / stride_h + 1;
    int ow = (w + 2 * q - dilate_filter_w) / stride_w + 1;

    int lds_size_approximate = (bacths_Single_block) * (group_single_block) *
                               ((h + 2 * p) * (w + 2 * q) + oh * ow) * sizeof(float);

    int lds = 256 * LDS_MAX_SIZE;

    printf("lds_size_approximate: %d\n", lds_size_approximate);
    printf("lds: %d\n", lds);

    /*
    1;
    192;
    181;
    380;
    192;
    7;
    11;
    3;
    5;
    1;
    1;
    1;
    1;
    192;
    lds_size_approximate: 9069440
    lds: 3840000


    1;
    192;
    182;
    380;
    384;
    2;
    2;
    0;
    0;
    2;
    2;
    1;
    1;
    1;
    lds_size_approximate: 5532800
    lds: 3840000

    1;
    79;
    181;
    380;
    192;
    1;
    1;
    0;
    0;
    1;
    1;
    1;
    1;
    1;
    lds_size_approximate: 8803840
    lds: 3840000
    */
}
