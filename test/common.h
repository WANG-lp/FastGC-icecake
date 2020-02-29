#include "../include/dlpack.h"

void dltensor_deleter_for_test(DLManagedTensor* tensor);
DLManagedTensor* make_DLM_tensor();

int cmp_dlm_tensor(const DLManagedTensor* tensor1, const DLManagedTensor* tensor2);