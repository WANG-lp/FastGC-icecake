#include <pybind11/pybind11.h>
#include "../include/icecake.hpp"

namespace py = pybind11;
using icecake::GPUCache;
using namespace pybind11::literals;
int add(int i, int j) { return i + j; }

namespace icecake {
bool GPUCache::put_dltensor(const string& fid, void* capsule) {
    return put_dltensor_to_device_memory(fid, (DLManagedTensor*) capsule);
}
py::capsule GPUCache::get_dltensor(const string& fid, int device) {
    return py::capsule(get_dltensor_from_device(fid, device), "dltensor");
}
}  // namespace icecake

PYBIND11_MODULE(pyicecake, m) {
    m.doc() = "pybind11 icecake plugin";
    m.def("add", &add, "A function which adds two numbers");
    py::class_<GPUCache>(m, "GPUCache")
        .def(py::init<size_t>())
        .def("put_dltensor", &GPUCache::put_dltensor, "Put a DLManagedTensor to device memory", "tensor_name"_a,
             "tensor"_a)
        .def("get_dltensor", &GPUCache::get_dltensor, "Get a DLManagedTensor from device memory", "tensor_name"_a,
             "deviceID"_a);
}
