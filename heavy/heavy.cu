#include <string.h>
#include <openssl/sha.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <map>

#ifndef _WIN32
#include <unistd.h>
#endif

// include thrust
/*
#include <thrust/version.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
*/
#include "miner2.h"



// Zahl der CUDA Devices im System bestimmen
extern "C" int cuda_num_devices()
{
    int version;
    cudaError_t err = cudaDriverGetVersion(&version);
    if (err != cudaSuccess)
    {
     //   applog(LOG_ERR, "Unable to query CUDA driver version! Is an nVidia driver installed?");
        exit(1);
    }

    int maj = version / 1000, min = version % 100; // same as in deviceQuery sample
    if (maj < 5 || (maj == 5 && min < 5))
    {
    //    applog(LOG_ERR, "Driver does not support CUDA %d.%d API! Update your nVidia driver!", 5, 5);
        exit(1);
    }

    int GPU_N;
    err = cudaGetDeviceCount(&GPU_N);
    if (err != cudaSuccess)
    {
     //   applog(LOG_ERR, "Unable to query number of CUDA devices! Is an nVidia driver installed?");
        exit(1);
    }
    return GPU_N;
}

// Gerätenamen holen
extern char *device_name[8];
extern int device_map[8];
int device_major[8];
int device_minor[8];

extern "C" void cuda_devicenames()
{
    cudaError_t err;
    int GPU_N;
    err = cudaGetDeviceCount(&GPU_N);
    if (err != cudaSuccess)
    {
     //   applog(LOG_ERR, "Unable to query number of CUDA devices! Is an nVidia driver installed?");
        exit(1);
    }

    for (int i=0; i < GPU_N; i++)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device_map[i]);

        device_name[i] = strdup(props.name);
		device_major[i] = props.major; 
		device_minor[i] = props.minor;
    }
}

extern "C" void cuda_deviceproperties(int GPU_N)
{
	for (int i = 0; i < GPU_N; i++)
	{
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, device_map[i]);

		device_name[i] = strdup(props.name);
		device_major[i] = props.major;
		device_minor[i] = props.minor;
	}
}

static bool substringsearch(const char *haystack, const char *needle, int &match)
{
    int hlen = strlen(haystack);
    int nlen = strlen(needle);
    for (int i=0; i < hlen; ++i)
    {
        if (haystack[i] == ' ') continue;
        int j=0, x = 0;
        while(j < nlen)
        {
            if (haystack[i+x] == ' ') {++x; continue;}
            if (needle[j] == ' ') {++j; continue;}
            if (needle[j] == '#') return ++match == needle[j+1]-'0';
            if (tolower(haystack[i+x]) != tolower(needle[j])) break;
            ++j; ++x;
        }
        if (j == nlen) return true;
    }
    return false;
}

// CUDA Gerät nach Namen finden (gibt Geräte-Index zurück oder -1)
extern "C" int cuda_finddevice(char *name)
{
    int num = cuda_num_devices();
    int match = 0;
    for (int i=0; i < num; ++i)
    {
        cudaDeviceProp props;
        if (cudaGetDeviceProperties(&props, i) == cudaSuccess)
            if (substringsearch(props.name, name, match)) return i;
    }
    return -1;
}

