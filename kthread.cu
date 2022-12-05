#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>
#include "kthread.h"
#include <cuda.h>

#if (defined(WIN32) || defined(_WIN32)) && defined(_MSC_VER)
#define __sync_fetch_and_add(ptr, addend)     _InterlockedExchangeAdd((void*)ptr, addend)
#endif

/************
 * kt_for() *
 ************/

struct kt_for_t;

typedef struct {
	struct kt_for_t *t;
	long i;
} ktf_worker_t;

typedef struct kt_for_t {
	int n_threads;
	long n;
	ktf_worker_t *w;
	void (*func)(void*,long,int);
	void *data;
} kt_for_t;

static inline long steal_work(kt_for_t *t)
{
	int i, min_i = -1;
	long k, min = LONG_MAX;
	for (i = 0; i < t->n_threads; ++i)
		if (min > t->w[i].i) min = t->w[i].i, min_i = i;
	k = __sync_fetch_and_add(&t->w[min_i].i, t->n_threads);
	return k >= t->n? -1 : k;
}

static void *ktf_worker(void *data)
{
	ktf_worker_t *w = (ktf_worker_t*)data;
	long i;
	for (;;) {
		i = __sync_fetch_and_add(&w->i, w->t->n_threads);
		if (i >= w->t->n) break;
		w->t->func(w->t->data, i, w - w->t->w);
	}
	while ((i = steal_work(w->t)) >= 0)
		w->t->func(w->t->data, i, w - w->t->w);
	pthread_exit(0);
}

void kt_for(int n_threads, void (*func)(void*,long,int), void *data, long n)
{
	if (n_threads > 1) {
		int i;
		kt_for_t t;
		pthread_t *tid;
		t.func = func, t.data = data, t.n_threads = n_threads, t.n = n;
		t.w = (ktf_worker_t*)calloc(n_threads, sizeof(ktf_worker_t));
		tid = (pthread_t*)calloc(n_threads, sizeof(pthread_t));
		for (i = 0; i < n_threads; ++i)
			t.w[i].t = &t, t.w[i].i = i;
		for (i = 0; i < n_threads; ++i) pthread_create(&tid[i], 0, ktf_worker, &t.w[i]);
		for (i = 0; i < n_threads; ++i) pthread_join(tid[i], 0);
		free(tid); free(t.w);
	} else {
		long j;
		for (j = 0; j < n; ++j) func(data, j, 0);
	}
}

/*****************
 * kt_pipeline() *
 *****************/

struct ktp_t;

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


//uses several threads, look into pthread.c
static void *ktp_worker(void *data)
{
	ktp_worker_t *w = (ktp_worker_t*)data; //creates a worker given data
	ktp_t *p = w->pl; //grabs the ktp struct from the worker
	while (w->step < p->n_steps) { // similar to if(n<N) from CUDA
		// test whether we can kick off the job with this worker
		pthread_mutex_lock(&p->mutex); //mutex is a synchronization lock to protect multiple thread access
		for (;;) {
			int i;
			// test whether another worker is doing the same step
			for (i = 0; i < p->n_workers; ++i) {
				if (w == &p->workers[i]) continue; // ignore itself
				if (p->workers[i].step <= w->step && p->workers[i].index < w->index)//if a different worker has a step less than or equal to ours, and their index is less than ours
					break;
			}
			if (i == p->n_workers) break; // no workers with smaller indices are doing w->step or the previous steps
			pthread_cond_wait(&p->cv, &p->mutex); //seems like it's similar to syncthreads
		}
		pthread_mutex_unlock(&p->mutex);

		// working on w->step
		//Consider editing this function to be a CUDA __device__ function?
		w->data = p->func(p->shared, w->step, w->step? w->data : 0); // for the first step, input is NULL
		//the data of the worker is done by using the function on shared data, for a specific step, and the step value seems to be null at first? I think this is because the 
		//data has yet to be filled, and step exists the final value in the function is NULL. For future steps, data won't be null.

		// update step and let other workers know
		pthread_mutex_lock(&p->mutex);
		w->step = w->step == p->n_steps - 1 || w->data? (w->step + 1) % p->n_steps : p->n_steps;
		//step = 1 if it's one less than total n_steps, second part is confusing but the ternary is saying 
		//if the data in the worker exists, increment step by one modulo 3, if data doesn't exist, 3
		//I have no idea what this has to do with the OR statement
		if (w->step == 0) w->index = p->index++; //if the step is 0, the worker index is set to the struct index + 1, which I believe means it fails the while loop next time around
		pthread_cond_broadcast(&p->cv);
		pthread_mutex_unlock(&p->mutex);
	}
	pthread_exit(0); 
}

//Data = a worker struct
//N = number of total workers 
__global__ void cuda_worker(int N, ktp_worker_t** workers)
{
	for(int i = 0; i < 200000000; i++){
		pow(128,25);
	}
	int t = threadIdx.x;
	int b = blockIdx.x;
	int B = blockDim.x;

	int n = t + b*B;
	if(n<N){
		ktp_worker_t *w = workers[n];
	
		ktp_t *p = w->pl;
		if(w->step < 3){
			w->data = p->func(p->shared, w->step, w->step? w->data : 0);
		}
		__syncthreads();
		w->step = w->step+1;
		if(w->step < 3){
			w->data = p->func(p->shared, w->step, w->step? w->data : 0);
		}
		__syncthreads();
		w->step = w->step+1;
		if(w->step < 3){
			w->data = p->func(p->shared, w->step, w->step? w->data : 0);
		}
		__syncthreads();
		
	}
}

float cuda_pipeline(int N, void *(*func)(void*, int, void*), void *shared_data, int n_steps, float time)
{
	ktp_t aux;
	cudaEvent_t start,stop;
	int i;
	if (N<1){N = 1;}
	aux.n_workers = N; //one worker per thread
	aux.n_steps = n_steps; //number of steps
	aux.func = func; //the function given
	aux.shared = shared_data; //maybe the data given to this pipeline to work on?
	aux.index = 0;
	aux.workers = (ktp_worker_t*)calloc(N, sizeof(ktp_worker_t)); //creates a worker for each thread and callocs an array
	for (i = 0; i < N; ++i) { //assigning index values to each worker
		ktp_worker_t *w = &aux.workers[i]; 
		w->step = 0; w->pl = &aux; w->data = 0; //w->pl allows the worker struct to reference the struct it's contained inside
		w->index = aux.index++; //increments the index each time through
	}
	cudaMalloc(&aux.c_workers, N*sizeof(ktp_worker_t));
	cudaMalloc(&aux.c_shared, N*sizeof(shared_data));
	cudaMemcpy(aux.c_workers, aux.workers, N*sizeof(ktp_worker_t),cudaMemcpyHostToDevice);
	cudaMemcpy(aux.c_shared, aux.shared, N*sizeof(shared_data), cudaMemcpyHostToDevice);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int B = 64;
	int G = (N+B-1)/B;
	cudaEventRecord(start);
	cuda_worker<<<G, B>>>(N, &aux.c_workers);

	cudaMemcpy(aux.workers, aux.c_workers, N*sizeof(ktp_worker_t),cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);
	cudaMemcpy(aux.shared, aux.c_shared, N*sizeof(shared_data), cudaMemcpyDeviceToHost);
	
	float elapsed;
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaFree(&aux.c_workers);
	free(aux.workers);
	return elapsed;
}

// Creates workers that then work together on doing the function defined by the worker_pipeline function
//n_threads = number of threads
//func = the function each worker will carry out
//shared_data = the pipeline struct pointer that is sent to the function
//n_steps = the number of steps for each thread
void kt_pipeline(int n_threads, void *(*func)(void*, int, void*), void *shared_data, int n_steps)
{
	ktp_t aux;
	pthread_t *tid;
	int i;

	if (n_threads < 1) n_threads = 1;
	aux.n_workers = n_threads; //one worker per thread
	aux.n_steps = n_steps; //number of steps
	aux.func = func; //the function given
	aux.shared = shared_data; //maybe the data given to this pipeline to work on?
	aux.index = 0;
	pthread_mutex_init(&aux.mutex, 0);
	pthread_cond_init(&aux.cv, 0);

	aux.workers = (ktp_worker_t*)calloc(n_threads, sizeof(ktp_worker_t)); //creates a worker for each thread and callocs an array
	for (i = 0; i < n_threads; ++i) { //assigning index values to each worker
		ktp_worker_t *w = &aux.workers[i]; 
		w->step = 0; w->pl = &aux; w->data = 0; //w->pl allows the worker struct to reference the struct it's contained inside
		w->index = aux.index++; //increments the index each time through
	}

	tid = (pthread_t*)calloc(n_threads, sizeof(pthread_t)); //tid is storing n thread objects
	for (i = 0; i < n_threads; ++i) pthread_create(&tid[i], 0, ktp_worker, &aux.workers[i]); //creating a thread with the index's thread object, that does the function "ktp_worker", using a specific worker
	for (i = 0; i < n_threads; ++i) pthread_join(tid[i], 0); //gets the returned values from the threads
	free(tid); free(aux.workers);

	pthread_mutex_destroy(&aux.mutex);
	pthread_cond_destroy(&aux.cv);
}
