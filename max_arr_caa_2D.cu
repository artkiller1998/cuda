#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h> 
#include <ctime>
#include <time.h> 

#define BLOCK_SIZE 1024

cudaError_t find_max(int *c, int *a, long long int size);


__global__ void find_kernel(int *c, int *a, long long int size)
{ 
	__shared__ int data [BLOCK_SIZE];
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x; 
	int index=gridDim.x*blockDim.x*blockIdx.y+blockIdx.x*blockDim.x+threadIdx.x;
	
	if (index < size)
	{
		data[tid] = a[index]; 
	
		__syncthreads (); 
		for ( int s = 1; s < blockDim.x; s *= 2 )  {
			if ( tid % (2*s) == 0 ) 
				if (data[tid] < data[tid + s])
					data [tid] = data [tid + s]; 
			__syncthreads (); 
		} 
		if ( tid == 0 ) 
		{
			c[blockIdx.x+blockIdx.y*gridDim.x] = data[0];
		}
	}
}

using namespace std;

int main(int argc, char* argv[])
{	
	
	long long int count = 0;
	string output = "";
	srand( time(0) );
	int max_val_c = 0;

	int limit = 0;

	if (argc == 7)
	{
		for (int i = 0; i < 7; i++)
		{
			if (!strcmp(argv[i], "--count"))
			{
				count = (long long int)atoi(argv[i + 1]);
			}
			if (!strcmp(argv[i], "--output"))
			{
				output = argv[i + 1];
			}
			if (!strcmp(argv[i], "--limit"))
			{
				limit = atoi(argv[i + 1]);
			}
		}
	}
	else
	{
		cout << "Enter correct parameters\n";
		cout << "Example: nvcc --run max_arr.cu -run-args \"--output result.txt --count 200 --limit 700\"";
		exit(1);
	}

	cout << "-- Input args --" ;
	cout << endl << "Count = " << count;
	cout << endl << "Output_file = " << output;
	cout << endl << "Limit = " << limit;
	cout << endl << "-- ========== --" << endl << endl;
	
	limit+=1;
	int *array = new int [count];
	int *res_array = new int [1];
	int i = 0;
	
	cout << "---  Array ----" << endl;
	unsigned int start_time_c =  clock();
	while (i < count)
	{
		array[i] = rand() % limit;
		if (max_val_c < array[i])
			max_val_c = array[i];

		i++;
	}
	unsigned int end_time_c = clock();
	unsigned int search_time_c = end_time_c - start_time_c;
	cout << endl << "---  ===== ----\n\n" ;
	
	cout << "---  CPU Results ----";
	printf("\nMax value: %lld\n", max_val_c );
	printf("Search value CPU: %.2f", (search_time_c / 1000000.0) );
	cout << endl << "---  =========== ----" << endl;


    //Add vectors in parallel.
	unsigned int start_time_g =  clock();
    cudaError_t cudaStatus = find_max(res_array, array, count);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "find_max failed!");
        return 1;
    }
	unsigned int end_time_g = clock();
	unsigned int search_time_g = end_time_g - start_time_g;
	
	cout << endl << "---  GPU Results ----";
	printf("\nMax value: %lld\n", res_array[0] );
	printf("Search value GPU: %.2f", (search_time_g / 1000000.0) );
	cout << endl << "---  =========== ----\n" << endl;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
	
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	
	ofstream out_file;
	out_file.open(output.c_str());
	if (!out_file.is_open()) {
		cerr << "\n\nAn error while openning file accured! Exiting...\n";
		return 1;
	}
	else {
		out_file << "---  CPU Results ----";
		out_file << "\nMax value: " << max_val_c << "\n";
		out_file << "Search value CPU: " << (search_time_c / 1000000.0) << "\n";
		out_file << "---  =========== ----" << endl;
		
		out_file << endl << "---  GPU Results ----";
		out_file << "\nMax value: " << res_array[0] << "\n";
		out_file << "Search value CPU: " << (search_time_g / 1000000.0) << "\n";
		out_file << "---  =========== ----\n" << endl;
	}
	out_file.close();
	
	delete res_array;
	delete array;
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t find_max(int *c, int *a, long long int size)
{
    int *dev_a = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;
	
	dim3 grid(BLOCK_SIZE, BLOCK_SIZE);
	dim3 block(BLOCK_SIZE);
	
	int BLOCK_CNT;
	BLOCK_CNT = ceil((long double)size / BLOCK_SIZE );
	//printf("\n1Block cnt is %d\n", BLOCK_CNT);
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	    
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "1cudaMemcpy failed!");
        goto Error;
    }
	
	
    // Launch a kernel on the GPU with one thread for each element.

	find_kernel<<<grid,block>>>(dev_c, dev_a, size);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "find_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	
	BLOCK_CNT = ceil((long double)BLOCK_CNT / BLOCK_SIZE );
	
	while (BLOCK_CNT != 1)
	{			
		find_kernel<<<grid,block>>>(dev_c, dev_c, size);
		
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "find_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		
		BLOCK_CNT = ceil((long double)BLOCK_CNT / (BLOCK_SIZE*BLOCK_SIZE) );
		//printf("\nnBlock cnt is %d\n", BLOCK_CNT);
	}
	
	find_kernel<<<grid,block>>>(dev_c, dev_c, size);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "find_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	//printf("\nnFinished%d\n", BLOCK_CNT);
    // Check for any errors launching the kernel

    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching find_kernel!\n", cudaStatus);
        goto Error;
    }

    //Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(a, dev_a, size*sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "2cudaMemcpy failed!");
        goto Error;
    }
	
	//Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, 1*sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "2cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_a);
	cudaFree(dev_c);
    
    return cudaStatus;
}
