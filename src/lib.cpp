#include "../include/lib.h"

#include <iostream>
#include <string>
#include <vector>
using std::cout;
using std::endl;
using std::string;
using std::vector;
int sub(int a, int b) {
  cout << "computing " << a << "-" << b << " in cpp code" << endl;
  if (a > b) {
    return a - b;
  } else {
    return b - a;
  }
}

char* csayhello(const char* str) {
  vector<string> ss = {"hello,", "world,", str};
  string out;
  for (const auto s : ss) {
    out.append(s);
  }

  char* cbuf = strdup(out.c_str());
  return cbuf;
}