#include <stdio.h>

int main(int argc, const char *argv[]) {
  char a = 0x80; // -128
  if (a == char(0x80)) {
    printf("Match\n");
  } else {
    printf("Mismatch\n");
  }
}
