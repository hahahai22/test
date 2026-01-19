#include <stdio.h>

extern void func();

extern int D_val;
extern const int R_val;

int main(int argc, char const *argv[])
{
  func();
  printf("D_val: %d\n", D_val);
  printf("R_val: %d\n", R_val);
  return 0;
}
