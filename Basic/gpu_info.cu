#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

int main() {
    int numberOfDevices;
    cudaGetDeviceCount(&numberOfDevices);
    printf("Number of CUDA-capable GPUs: %d\n", numberOfDevices);

    for (int i = 0; i < numberOfDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("\n=================== Device %d ===================\n", i);
        printf("Name: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Clock rate: %.0f MHz (%.2f GHz)\n", prop.clockRate * 1e-3f, prop.clockRate * 1e-6f);
        printf("Global memory: %.0f MB (%llu bytes)\n", prop.totalGlobalMem / 1048576.0f, (unsigned long long)prop.totalGlobalMem);
        printf("Shared memory per block: %zu bytes\n", (size_t)prop.sharedMemPerBlock);
        printf("Shared memory per multiprocessor: %zu bytes\n", (size_t)prop.sharedMemPerMultiprocessor);
        printf("Constant memory: %zu bytes\n", (size_t)prop.totalConstMem);
        printf("Registers per block: %d\n", prop.regsPerBlock);
        printf("Registers per multiprocessor: %d\n", prop.regsPerMultiprocessor);
        printf("Warp size: %d\n", prop.warpSize);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Number of multiprocessors (SMs): %d\n", prop.multiProcessorCount);
        printf("Max block dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("Memory Clock Rate: %.0f MHz\n", prop.memoryClockRate * 1e-3f);
        printf("Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
        printf("Texture Alignment: %zu bytes\n", (size_t)prop.textureAlignment);
        printf("Device Overlap (Async copy + kernel): %s\n", prop.deviceOverlap ? "Supported" : "Not supported");

        // ویژگی‌های پیشرفته‌تر:
        printf("Can map host memory: %s\n", prop.canMapHostMemory ? "Yes" : "No");
        printf("Concurrent kernel execution: %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("ECC enabled: %s\n", prop.ECCEnabled ? "Yes" : "No");
        printf("Integrated GPU (shared with CPU): %s\n", prop.integrated ? "Yes" : "No");
        printf("Multi-GPU board: %s\n", prop.isMultiGpuBoard ? "Yes" : "No");

        printf("PCI Bus ID: %d\n", prop.pciBusID);
        printf("PCI Device ID: %d\n", prop.pciDeviceID);
        printf("Async Engine Count: %d\n", prop.asyncEngineCount);
        printf("L2 Cache Size: %d bytes\n", prop.l2CacheSize);

        printf("Managed memory supported: %s\n", prop.managedMemory ? "Yes" : "No");
        printf("Pageable memory access: %s\n", prop.pageableMemoryAccess ? "Yes" : "No");
        printf("Concurrent managed access: %s\n", prop.concurrentManagedAccess ? "Yes" : "No");
    }

    return 0;
}
