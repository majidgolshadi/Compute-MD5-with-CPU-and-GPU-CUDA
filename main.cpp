/**
* majid Golshadi
* CUDA version: 5.5
*/

#include <string>
#include <stdio.h>
#include <ctime>
#include "md5kernel.h"
#include "md5.h"
#include "cpuMD5.h"


int main(int argc, char **argv) {
    char *msg = argv[1];
    size_t len;
    int i;
	long b = 1000;
    uint8_t result[16];
	cudaError_t cudaStatus;
	clock_t begin, end, delta;
	double timeSec;
 
    if (argc < 2) {
        printf("usage: %s 'string'\n", argv[0]);
        return 1;
    }

	printf("\n\n");
	/*************************************************************************/
	
	len = strlen(msg);
	begin = clock();
    for (i = 0; i < b; i++) {
        md5((uint8_t*)msg, len, result);
    }
	end = clock();

	// display result 
	std::cout<<"MD5 result of "<<argv[1]<<" is: ";
    for (i = 0; i < 16; i++)
        printf("%2.2x", result[i]);
	std::cout<<std::endl<<"and compute times are as follows "<<std::endl<<std::endl;

	timeSec = static_cast<float>(b / ((end - begin) / static_cast<float>(CLOCKS_PER_SEC)));
	printf("CPU v1: ");
	printf("%fh/s\n", timeSec);

	/*************************************************************************/
	begin = clock();
	std::cout<<begin<<std::endl;
	std::string input = argv[1];
	std::string resultV2 = cpuMd5(input);
	end = clock();
	std::cout<<end<<std::endl;
	std::cout<<end - begin<<std::endl;
	timeSec = static_cast<float>(b / ((end - begin) / static_cast<float>(CLOCKS_PER_SEC)));
	printf("CPU v2: ");
	printf("%fh/s\n\n", timeSec);

	puts("");
	/*************************************************************************/
	printf("GPU V1 (with memory transfers timed): ");
	 

	begin = clock();
    for (i = 0; i < b; i++) {
		cudaStatus = md5WithCuda((uint8_t*)msg, len, result);
		if(cudaStatus != cudaSuccess) {
			printf("An error with CUDA occured!\n");
			break;
		}
    }
	end = clock();

	if(cudaStatus == cudaSuccess) {
		timeSec = static_cast<float>(b / ((end - begin) / static_cast<float>(CLOCKS_PER_SEC)));
		printf("%fh/s\n", timeSec);
	} else {
		printf("CUDA timing invalid because of error\n");
	}

	/*************************************************************************/
	printf("GPU V2(without memory transfers timed): ");
    
	delta = 0;
    for (i = 0; i < b; i++) {
		cudaStatus = md5WithCudaTimed((uint8_t*)msg, len, result, begin, end);
		if(cudaStatus != cudaSuccess) {
			printf("An error with CUDA occured!\n");
			break;
		} else {
			delta += end - begin;
		}
    }
	if(cudaStatus == cudaSuccess) {
		timeSec = static_cast<float>(b / (delta / static_cast<float>(CLOCKS_PER_SEC)));
		printf("%fh/s\n", timeSec);
	} else {
		printf("CUDA timing invalid because of error\n");
	}

	/*************************************************************************/
	printf("GPU V3(with memory transfers timed but kernel looped): ");
	begin = clock();
    // benchmark gpu
	cudaStatus = md5WithCudaRounds((uint8_t*)msg, len, result, b);
	if(cudaStatus != cudaSuccess) {
		printf("An error with CUDA occured!\n");
	}
	end = clock();
	if(cudaStatus == cudaSuccess) {
		timeSec = static_cast<float>(b / ((end - begin) / static_cast<float>(CLOCKS_PER_SEC)));
		printf("%fh/s\n", timeSec);
	} else {
		printf("CUDA timing invalid because of error\n");
	}

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    
    return 0;
}