#include <pybind11/pybind11.h>

#include "dense_decode.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    m.def("dense_decode_fwd", &dense_attn_decode_interface);
}
