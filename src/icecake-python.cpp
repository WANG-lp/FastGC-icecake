#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <iostream>
#include "../include/icecake.hpp"
#include "utils/chromiumbase64.h"

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
py::array GPUCache::get_numpy_array(const string& fid) {
    auto cap = get_dltensor_from_device(fid, 0);
    std::vector<ssize_t> shapes;
    std::vector<ssize_t> strides;
    for (int i = 0; i < cap->dl_tensor.ndim; i++) {
        shapes.push_back(cap->dl_tensor.shape[i]);
        strides.push_back(cap->dl_tensor.strides[i]);
    }
    void* data = cap->dl_tensor.data;
    py::array ret;
    cap->deleter(cap);
    if (cap->dl_tensor.dtype.code == DLDataTypeCode::kDLBfloat && cap->dl_tensor.dtype.bits == 16) {
        ret = py::array(py::dtype("float16"), shapes, strides, data);
    } else if (cap->dl_tensor.dtype.code == DLDataTypeCode::kDLFloat && cap->dl_tensor.dtype.bits == 32) {
        return py::array(py::dtype("float32"), shapes, strides, data);
    } else if (cap->dl_tensor.dtype.code == DLDataTypeCode::kDLInt) {
        if (cap->dl_tensor.dtype.bits == 8) {
            return py::array(py::dtype("int8"), shapes, strides, data);
        } else if (cap->dl_tensor.dtype.bits == 16) {
            return py::array(py::dtype("int16"), shapes, strides, data);
        } else if (cap->dl_tensor.dtype.bits == 32) {
            return py::array(py::dtype("int32"), shapes, strides, data);
        } else if (cap->dl_tensor.dtype.bits == 64) {
            return py::array(py::dtype("int64"), shapes, strides, data);
        }
    } else if (cap->dl_tensor.dtype.code == DLDataTypeCode::kDLUInt) {
        if (cap->dl_tensor.dtype.bits == 8) {
            return py::array(py::dtype("uint8"), shapes, strides, data);
        } else if (cap->dl_tensor.dtype.bits == 16) {
            return py::array(py::dtype("uint16"), shapes, strides, data);
        } else if (cap->dl_tensor.dtype.bits == 32) {
            return py::array(py::dtype("uint32"), shapes, strides, data);
        } else if (cap->dl_tensor.dtype.bits == 64) {
            return py::array(py::dtype("uint64"), shapes, strides, data);
        }
    }
    spdlog::error("Incorrect dtype");
    return ret;
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

void GPUCache::save_dltensor_to_file(const string& fid, const string& fname) {
    auto cap = get_dltensor_from_device(fid, 0);
    if (cap == nullptr) {
        spdlog::error("cannot find {} dltensor", fid);
        return;
    }
    auto tensor = cap->dl_tensor;
    std::ofstream ofile(fname, std::ios::out | std::ios::binary);
    ofile << fid << std::endl;
    std::vector<char> data_buff = serialize_dl_tensor(&tensor);
    for (int i = 0; i < calc_dltensor_size(&tensor); i++) {
        data_buff.push_back(((char*) tensor.data)[i]);
    }
    std::vector<char> base64_buff;

    base64_buff.resize(data_buff.size() * 2);  // 2 times length is safe
    size_t len = chromium_base64_encode(base64_buff.data(), (const char*) data_buff.data(), data_buff.size());
    base64_buff.resize(len);  // shrink
    ofile << std::string(base64_buff.begin(), base64_buff.end()) << std::endl;
    ofile.close();
}
void GPUCache::load_dltensor_from_file(const string& fname) {
    std::ifstream ifile(fname, std::ios::in | std::ios::binary);
    if (!ifile.good()) {
        spdlog::error("cannot open file {}", fname);
        return;
    }
    std::string tname;
    DLManagedTensor* dlm_tensor = (DLManagedTensor*) malloc(sizeof(DLManagedTensor));
    ifile >> tname;
    std::string data_str;
    ifile >> data_str;
    vector<char> decoded_buf;
    decoded_buf.resize(data_str.size());
    auto len = chromium_base64_decode(decoded_buf.data(), data_str.data(), data_str.size());
    decoded_buf.resize(len);
    dlm_tensor->dl_tensor = deserialize_dl_tensor(decoded_buf.data());
    dlm_tensor->dl_tensor.ctx.device_type = DLDeviceType::kDLCPU;
    dlm_tensor->deleter = nullptr;
    put_dltensor_to_device_memory(tname, dlm_tensor);
    free(dlm_tensor);
}
size_t GPUCache::get_self_pointer_addr() {
    void* addr = this;
    return reinterpret_cast<std::uintptr_t>(addr);
}

}  // namespace icecake

PYBIND11_MODULE(pyicecake, m) {
    m.doc() = "pybind11 icecake plugin";
    m.def("add", &add, "A function which adds two numbers");
    py::class_<GPUCache>(m, "GPUCache")
        .def(py::init<size_t>())
        .def("get_self_pointer", &GPUCache::get_self_pointer_addr, "Get the pointer of this instance")
        .def("put_numpy_array", &GPUCache::put_numpy_array, "Put a Numpy Array to device memory", "tensor_name"_a,
             "narray"_a)
        .def("get_numpy_array", &GPUCache::get_numpy_array, "Get a Numpy Array from device memory", "tensor_name"_a)
        .def("put_dltensor", &GPUCache::put_dltensor, "Put a DLManagedTensor to device memory", "tensor_name"_a,
             "tensor"_a)
        .def("get_dltensor", &GPUCache::get_dltensor, "Get a DLManagedTensor from device memory", "tensor_name"_a,
             "deviceID"_a)
        .def("save_dltensor_to_file", &GPUCache::save_dltensor_to_file, "Save dltensor to a file", "tensor_name"_a,
             "output_file_name"_a)
        .def("load_dltensor_from_file", &GPUCache::load_dltensor_from_file, "Load dltensor from a file", "file_name"_a);
}
