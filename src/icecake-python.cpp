#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "../include/icecake.hpp"

namespace py = pybind11;
using icecake::GPUCache;
using namespace pybind11::literals;
int add(int i, int j) { return i + j; }

constexpr const char* DLTENSOR_NAME = "dltensor";
constexpr const char* USED_DLTENSOR_NAME = "used_dltensor";
namespace icecake {
static bool set_dtype(char dtype_name, size_t dtype_size, DLTensor* tensor) {
    tensor->dtype.lanes = 1;
    tensor->dtype.bits = dtype_size * 8;

    if (dtype_name == 'f' && dtype_size == 2) {
        tensor->dtype.code = DLDataTypeCode::kDLBfloat;
    } else if (dtype_name == 'f' && dtype_size == 4) {
        tensor->dtype.code = DLDataTypeCode::kDLFloat;
    } else if (dtype_name == 'u') {
        tensor->dtype.code = DLDataTypeCode::kDLUInt;
    } else if (dtype_name == 'i') {
        tensor->dtype.code = DLDataTypeCode::kDLInt;
    } else {
        spdlog::error("Incorrect dtype: {}, size: {}", dtype_name, dtype_size);
        return false;
    }
    return true;
}
static void DLM_tensor_capsule_destructor(PyObject* capsule) {
    if (strcmp(PyCapsule_GetName(capsule), DLTENSOR_NAME) == 0) {
        auto* ptr = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, DLTENSOR_NAME));
        dltensor_deleter((DLManagedTensor*) capsule);
    }
}
bool GPUCache::put_numpy_array(const string& fid, py::array narray) {
    DLManagedTensor dlm_tensor;
    dlm_tensor.deleter = nullptr;
    dlm_tensor.dl_tensor.ctx.device_id = 0;
    dlm_tensor.dl_tensor.ctx.device_type = DLDeviceType::kDLCPU;
    dlm_tensor.dl_tensor.byte_offset = 0;
    dlm_tensor.dl_tensor.data = (void*) narray.data();
    if (!set_dtype(narray.dtype().kind(), narray.dtype().itemsize(), &(dlm_tensor.dl_tensor))) {
        return false;
    }
    dlm_tensor.dl_tensor.ndim = narray.ndim();
    dlm_tensor.dl_tensor.shape = (int64_t*) narray.shape();
    dlm_tensor.dl_tensor.strides = (int64_t*) narray.strides();
    return put_dltensor_to_device_memory(fid, &dlm_tensor);
}

bool GPUCache::put_dltensor(const string& fid, py::capsule capsule) {
    if (strcmp(capsule.name(), DLTENSOR_NAME) != 0) {
        spdlog::error("A dltensor can be consumed only once!");
        return false;
    }
    PyCapsule_SetName(capsule.ptr(), USED_DLTENSOR_NAME);
    return put_dltensor_to_device_memory(fid, (DLManagedTensor*) capsule);
}
py::capsule GPUCache::get_dltensor(const string& fid, int device) {
    auto cap = get_dltensor_from_device(fid, device);
    // char* data = cap->dl_tensor.data;
    return py::capsule(cap, "dltensor", &DLM_tensor_capsule_destructor);
}
}  // namespace icecake

PYBIND11_MODULE(pyicecake, m) {
    m.doc() = "pybind11 icecake plugin";
    m.def("add", &add, "A function which adds two numbers");
    py::class_<GPUCache>(m, "GPUCache")
        .def(py::init<size_t>())
        .def("put_numpy_array", &GPUCache::put_numpy_array, "Put a Numpy Array to device memory", "tensor_name"_a,
             "narray"_a)
        .def("put_dltensor", &GPUCache::put_dltensor, "Put a DLManagedTensor to device memory", "tensor_name"_a,
             "tensor"_a)
        .def("get_dltensor", &GPUCache::get_dltensor, "Get a DLManagedTensor from device memory", "tensor_name"_a,
             "deviceID"_a);
}
