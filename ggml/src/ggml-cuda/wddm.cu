#include "wddm.cuh"

#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <dxgi1_4.h>

#include <cstring>
#include <mutex>

#pragma comment(lib, "dxgi.lib")

bool ggml_cuda_wddm_query(const char * luid8, uint64_t * budget, uint64_t * dedicated, uint64_t * shared) {
    static std::mutex      mutex;
    static IDXGIAdapter3 * adapter = nullptr;
    static char            adapter_luid[8] = {};
    static bool            tried = false;

    std::lock_guard<std::mutex> lock(mutex);
    if (!tried || memcmp(adapter_luid, luid8, 8) != 0) {
        tried = true;
        memcpy(adapter_luid, luid8, 8);
        if (adapter != nullptr) {
            adapter->Release();
            adapter = nullptr;
        }
        IDXGIFactory4 * factory = nullptr;
        if (SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void **) &factory)) && factory != nullptr) {
            for (UINT i = 0; adapter == nullptr; ++i) {
                IDXGIAdapter1 * a = nullptr;
                if (factory->EnumAdapters1(i, &a) != S_OK || a == nullptr) {
                    break;
                }
                DXGI_ADAPTER_DESC1 desc;
                if (SUCCEEDED(a->GetDesc1(&desc)) && memcmp(&desc.AdapterLuid, luid8, 8) == 0) {
                    a->QueryInterface(__uuidof(IDXGIAdapter3), (void **) &adapter);
                }
                a->Release();
            }
            factory->Release();
        }
    }
    if (adapter == nullptr) {
        return false;
    }
    DXGI_QUERY_VIDEO_MEMORY_INFO local = {}, non_local = {};
    if (FAILED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local)) ||
        FAILED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &non_local))) {
        return false;
    }
    *budget    = local.Budget;
    *dedicated = local.CurrentUsage;
    *shared    = non_local.CurrentUsage;
    return true;
}

#else

bool ggml_cuda_wddm_query(const char * luid8, uint64_t * budget, uint64_t * dedicated, uint64_t * shared) {
    (void) luid8; (void) budget; (void) dedicated; (void) shared;
    return false;
}

#endif
