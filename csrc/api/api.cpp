#include <pybind11/pybind11.h>

#include "dense_decode.h"
#include "dense_fwd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    m.def("dense_decode_fwd", &dense_attn_decode_interface);
    m.def("dense_prefill_fwd", &FMHACutlassSM100FwdRun);
    m.def("dense_prefill_bwd", &FMHACutlassSM100BwdRun);
}
