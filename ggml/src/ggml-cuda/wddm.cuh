#pragma once

#include <cstdint>

// Windows (WDDM) memory figures of THIS process on the adapter with the given LUID (cuDeviceGetLuid):
//   budget     dedicated bytes the system lets this process use now
//   dedicated  dedicated bytes in use
//   shared     bytes of this process's GPU memory that are in system RAM. Pinned host memory that the device maps by
//              design (the host tail of a tiered K/V cache, the exact rows of the two-stage head) counts here, and so
//              does memory that the driver's system-memory fallback moved out of VRAM (a "spill": reads get slow).
// Returns false on other systems, or when the adapter is not found.
bool ggml_cuda_wddm_query(const char * luid8, uint64_t * budget, uint64_t * dedicated, uint64_t * shared);
