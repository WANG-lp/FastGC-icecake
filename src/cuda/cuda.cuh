#pragma once

__global__ void _cusub(int* a, int *b, int *c);

int sub_cuda(int a, int b);