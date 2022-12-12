#ifndef MAP_CUH
#define MAP_CUH

#ifdef __cplusplus
extern "C" {
#endif

__device__ static void *worker_caller(void *shared, int step, void *in);
__device__ static void *worker_p0(void *shared, int step, void *in);
__device__ static void *worker_p1(void *shared, int step, void *in);
__device__ static void *worker_p2(void *shared, int step, void *in);

#ifdef __cplusplus
}
#endif

#endif
