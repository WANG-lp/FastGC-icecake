#include <pybind11/pybind11.h>
#include "../include/icecake.hpp"

namespace py = pybind11;
using icecake::GPUCache;
using namespace pybind11::literals;
int add(int i, int j) { return i + j; }

constexpr const char* DLTENSOR_NAME = "dltensor";
constexpr const char* USED_DLTENSOR_NAME = "used_dltensor";

namespace icecake {

static void DLM_tensor_capsule_destructor(PyObject* capsule) {
    if (strcmp(PyCapsule_GetName(capsule), DLTENSOR_NAME) == 0) {
        auto* ptr = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, DLTENSOR_NAME));
        dltensor_deleter((DLManagedTensor*) capsule);
    }
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
        .def("put_dltensor", &GPUCache::put_dltensor, "Put a DLManagedTensor to device memory", "tensor_name"_a,
             "tensor"_a)
        .def("get_dltensor", &GPUCache::get_dltensor, "Get a DLManagedTensor from device memory", "tensor_name"_a,
             "deviceID"_a);
}
