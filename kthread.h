#ifndef KTHREAD_CUH
#define KTHREAD_CUH

#ifdef __cplusplus
extern "C" {
#endif

float cuda_pipeline(int N, void *(*func)(void*, int, void*), void *shared_data, int n_steps, float elapsed);
void kt_for(int n_threads, void (*func)(void*,long,int), void *data, long n);
void kt_pipeline(int n_threads, void *(*func)(void*, int, void*), void *shared_data, int n_steps);

#ifdef __cplusplus
}
#endif

#endif
