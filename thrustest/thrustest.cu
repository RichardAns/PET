// thrustest
#pragma warning( disable : 4244)   // thrust::reduce int mismatch
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "thrust/execution_policy.h"
#include "thrust/for_each.h"
#include "thrust/scan.h"

#include <stdio.h>

struct printf_functor
{
	__host__ __device__  void operator()(int x)	{ printf("%d\n", x); }
};

__global__ void init_a(int *a){
    int id = blockIdx.x*blockDim.x + threadIdx.x;
	a[id] = id+1;
    
}

__global__ void scan_a(int *a){
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	a[id] = (id+1)%3;

}


int main(int argc,char *argv[])
{
	int threads = atoi(argv[1]);
	int blocks = atoi(argv[2]);
	int size = threads*blocks;

	thrust::host_vector<int>       a(size);
	thrust::device_vector<int> dev_a(size);

	init_a<<<blocks,threads>>>(dev_a.data().get());
	a = dev_a;

	int sum1 = thrust::reduce(a.begin(),a.end());
	printf("sum 1 %d\n",sum1);

	int sum2 = thrust::reduce(dev_a.begin(),dev_a.end());
	printf("sum 2 %d\n",sum2);

	// print without copy to host!
	thrust::for_each(thrust::device, dev_a.begin(), dev_a.end(), printf_functor());


	scan_a<<<blocks,threads>>>(dev_a.data().get());
	a = dev_a;
	for(int k=0;k<size;k++) printf(" %d", a[k]); printf("\n");

	// exclusice scan in place
	//thrust::exclusive_scan( thrust::device, dev_a.begin(), dev_a.end(), dev_a.begin()); // in-place scan?
	thrust::inclusive_scan( thrust::device, dev_a.begin(), dev_a.end(), dev_a.begin()); // in-place scan?
	a = dev_a;
	for(int k=0;k<size;k++) printf(" %d", a[k]); printf("\n");

    return 0;
}

