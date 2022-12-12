#ifndef KTHREAD_CUH
#define KTHREAD_CUH

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
	struct ktp_t *pl;
	int64_t index;
	int step;
	void *data;
} ktp_worker_t;

typedef struct ktp_t {
	void *shared;
	void *c_shared;
	void *(*func)(void*, int, void*);
	int64_t index;
	int n_workers, n_steps;
	ktp_worker_t *workers;
	ktp_worker_t *c_workers;
	pthread_mutex_t mutex;
	pthread_cond_t cv;
} ktp_t;


void cuda_worker(int N, ktp_worker_t** workers, void *shared_data);
void cuda_pipeline(int N, void *(*func)(void*, int, void*), void *shared_data, int n_steps);
void kt_for(int n_threads, void (*func)(void*,long,int), void *data, long n);
void kt_pipeline(int n_threads, void *(*func)(void*, int, void*), void *shared_data, int n_steps);

#ifdef __cplusplus
}
#endif

#endif
