#include "md5kernel.h"
#include "device_launch_parameters.h"

#include <stdio.h>

uint32_t *dev_k = 0;
uint32_t *dev_r = 0;

__device__ void cuda_to_bytes(uint32_t val, uint8_t *bytes)
{
    bytes[0] = (uint8_t) val;
    bytes[1] = (uint8_t) (val >> 8);
    bytes[2] = (uint8_t) (val >> 16);
    bytes[3] = (uint8_t) (val >> 24);
}
 
__device__ uint32_t cuda_to_int32(const uint8_t *bytes)
{
    return (uint32_t) bytes[0]
        | ((uint32_t) bytes[1] << 8)
        | ((uint32_t) bytes[2] << 16)
        | ((uint32_t) bytes[3] << 24);
}
 
__global__ void md5kernel(const uint8_t *initial_msg, size_t initial_len, uint8_t *digest, const uint32_t *k, const uint32_t *r) {
 
    // These vars will contain the hash
    uint32_t h0, h1, h2, h3;
 
    // Message (to prepare)
    uint8_t *msg = NULL;
 
    size_t new_len, offset;
    uint32_t w[16];
    uint32_t a, b, c, d, i, f, g, temp;
 
    // Initialize variables - simple count in nibbles:
    h0 = 0x67452301;
    h1 = 0xefcdab89;
    h2 = 0x98badcfe;
    h3 = 0x10325476;
 
    //Pre-processing:
    //append "1" bit to message    
    //append "0" bits until message length in bits ≡ 448 (mod 512)
    //append length mod (2^64) to message
 
    for (new_len = initial_len + 1; new_len % (512/8) != 448/8; new_len++)
        ;
 
    msg = (uint8_t *)malloc(new_len + 8);
    memcpy(msg, initial_msg, initial_len);
    msg[initial_len] = 0x80; // append the "1" bit; most significant bit is "first"
    for (offset = initial_len + 1; offset < new_len; offset++)
        msg[offset] = 0; // append "0" bits
 
    // append the len in bits at the end of the buffer.
    cuda_to_bytes(initial_len*8, msg + new_len);
    // initial_len>>29 == initial_len*8>>32, but avoids overflow.
    cuda_to_bytes(initial_len>>29, msg + new_len + 4);
 
    // Process the message in successive 512-bit chunks:
    //for each 512-bit chunk of message:
    for(offset=0; offset<new_len; offset += (512/8)) {
 
        // break chunk into sixteen 32-bit words w[j], 0 ≤ j ≤ 15
        for (i = 0; i < 16; i++)
            w[i] = cuda_to_int32(msg + offset + i*4);
 
        // Initialize hash value for this chunk:
        a = h0;
        b = h1;
        c = h2;
        d = h3;
 
        // Main loop:
        for(i = 0; i<64; i++) {
 
            if (i < 16) {
                f = (b & c) | ((~b) & d);
                g = i;
            } else if (i < 32) {
                f = (d & b) | ((~d) & c);
                g = (5*i + 1) % 16;
            } else if (i < 48) {
                f = b ^ c ^ d;
                g = (3*i + 5) % 16;          
            } else {
                f = c ^ (b | (~d));
                g = (7*i) % 16;
            }
 
            temp = d;
            d = c;
            c = b;
            b = b + LEFTROTATE((a + f + k[i] + w[g]), r[i]);
            a = temp;
 
        }
 
        // Add this chunk's hash to result so far:
        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
 
    }
 
    // cleanup
    free(msg);
 
    //var char digest[16] := h0 append h1 append h2 append h3 //(Output is in little-endian)
    cuda_to_bytes(h0, digest);
    cuda_to_bytes(h1, digest + 4);
    cuda_to_bytes(h2, digest + 8);
    cuda_to_bytes(h3, digest + 12);
}

__global__ void md5kernelRounds(const uint8_t *initial_msg, size_t initial_len, uint8_t *digest, const uint32_t *k, const uint32_t *r, const long rounds) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	for(int in = 0, l = rounds / blockDim.x; in < l; ++in) {
		// These vars will contain the hash
		uint32_t h0, h1, h2, h3;
 
		// Message (to prepare)
		uint8_t *msg = NULL;
 
		size_t new_len, offset;
		uint32_t w[16];
		uint32_t a, b, c, d, i, f, g, temp;
 
		// Initialize variables - simple count in nibbles:
		h0 = 0x67452301;
		h1 = 0xefcdab89;
		h2 = 0x98badcfe;
		h3 = 0x10325476;
 
		//Pre-processing:
		//append "1" bit to message    
		//append "0" bits until message length in bits ≡ 448 (mod 512)
		//append length mod (2^64) to message
 
		for (new_len = initial_len + 1; new_len % (512/8) != 448/8; new_len++)
			;
 
		msg = (uint8_t *)malloc(new_len + 8);
		memcpy(msg, initial_msg, initial_len);
		msg[initial_len] = 0x80; // append the "1" bit; most significant bit is "first"
		for (offset = initial_len + 1; offset < new_len; offset++)
			msg[offset] = 0; // append "0" bits
 
		// append the len in bits at the end of the buffer.
		cuda_to_bytes(initial_len*8, msg + new_len);
		// initial_len>>29 == initial_len*8>>32, but avoids overflow.
		cuda_to_bytes(initial_len>>29, msg + new_len + 4);
 
		// Process the message in successive 512-bit chunks:
		//for each 512-bit chunk of message:
		for(offset=0; offset<new_len; offset += (512/8)) {
 
			// break chunk into sixteen 32-bit words w[j], 0 ≤ j ≤ 15
			for (i = 0; i < 16; i++)
				w[i] = cuda_to_int32(msg + offset + i*4);
 
			// Initialize hash value for this chunk:
			a = h0;
			b = h1;
			c = h2;
			d = h3;
 
			// Main loop:
			for(i = 0; i<64; i++) {
 
				if (i < 16) {
					f = (b & c) | ((~b) & d);
					g = i;
				} else if (i < 32) {
					f = (d & b) | ((~d) & c);
					g = (5*i + 1) % 16;
				} else if (i < 48) {
					f = b ^ c ^ d;
					g = (3*i + 5) % 16;          
				} else {
					f = c ^ (b | (~d));
					g = (7*i) % 16;
				}
 
				temp = d;
				d = c;
				c = b;
				b = b + LEFTROTATE((a + f + k[i] + w[g]), r[i]);
				a = temp;
 
			}
 
			// Add this chunk's hash to result so far:
			h0 += a;
			h1 += b;
			h2 += c;
			h3 += d;
 
		}
 
		// cleanup
		free(msg);
		if(idx == 0) {
			//var char digest[16] := h0 append h1 append h2 append h3 //(Output is in little-endian)
			cuda_to_bytes(h0, digest);
			cuda_to_bytes(h1, digest + 4);
			cuda_to_bytes(h2, digest + 8);
			cuda_to_bytes(h3, digest + 12);
		}
	}
}

// Helper function for using CUDA to compute MD5
cudaError_t md5WithCuda(const uint8_t *initial_msg, size_t initial_len, uint8_t *digest)
{
    uint8_t *dev_initial_msg = 0;
    uint8_t *dev_digest = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output).
    cudaStatus = cudaMalloc((void**)&dev_k, k_size * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	
    cudaStatus = cudaMalloc((void**)&dev_r, r_size * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_digest, md5_size * sizeof(uint8_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_initial_msg, initial_len * sizeof(uint8_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_initial_msg, initial_msg, initial_len * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_k, cuda_k, k_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
    cudaStatus = cudaMemcpy(dev_r, cuda_r, r_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	md5kernel<<<1, initial_len>>>(dev_initial_msg, initial_len, dev_digest, dev_k, dev_r);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(digest, dev_digest, md5_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_digest);
    cudaFree(dev_initial_msg);
	cudaFree(dev_k);
	cudaFree(dev_r);
    
    return cudaStatus;
}

// Helper function for using CUDA to compute MD5 for several rounds
cudaError_t md5WithCudaRounds(const uint8_t *initial_msg, size_t initial_len, uint8_t *digest, const long rounds)
{
    uint8_t *dev_initial_msg = 0;
    uint8_t *dev_digest = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output).
    cudaStatus = cudaMalloc((void**)&dev_k, k_size * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	
    cudaStatus = cudaMalloc((void**)&dev_r, r_size * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_digest, md5_size * sizeof(uint8_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_initial_msg, initial_len * sizeof(uint8_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_initial_msg, initial_msg, initial_len * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_k, cuda_k, k_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
    cudaStatus = cudaMemcpy(dev_r, cuda_r, r_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	md5kernelRounds<<<512, initial_len>>>(dev_initial_msg, initial_len, dev_digest, dev_k, dev_r, rounds);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(digest, dev_digest, md5_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_digest);
    cudaFree(dev_initial_msg);
	cudaFree(dev_k);
	cudaFree(dev_r);
    
    return cudaStatus;
}

// Helper function for using CUDA to compute MD5 with timing
cudaError_t md5WithCudaTimed(const uint8_t *initial_msg, size_t initial_len, uint8_t *digest, clock_t& begin, clock_t& end)
{
    uint8_t *dev_initial_msg = 0;
    uint8_t *dev_digest = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output).
    cudaStatus = cudaMalloc((void**)&dev_k, k_size * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	
    cudaStatus = cudaMalloc((void**)&dev_r, r_size * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_digest, md5_size * sizeof(uint8_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_initial_msg, initial_len * sizeof(uint8_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_initial_msg, initial_msg, initial_len * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_k, cuda_k, k_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
    cudaStatus = cudaMemcpy(dev_r, cuda_r, r_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	// Start timing
	begin = clock();

    // Launch a kernel on the GPU with one thread for each element.
	md5kernel<<<1, initial_len>>>(dev_initial_msg, initial_len, dev_digest, dev_k, dev_r);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		// End timing
		end = clock();
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		// End timing
		end = clock();
        goto Error;
    }

	// End timing
	end = clock();

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(digest, dev_digest, md5_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_digest);
    cudaFree(dev_initial_msg);
	cudaFree(dev_k);
	cudaFree(dev_r);
    
    return cudaStatus;
}
