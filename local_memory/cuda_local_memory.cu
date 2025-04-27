#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const* func, char const* file, int line)
{
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA Runtime Error at: " << file << ":" << line
			<< std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

	template <int WindowSize>
__global__ void running_mean_register_array(float const* input, float* output,
		int n)
{
	float window[WindowSize];
	int const thread_idx{
		static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
	int const stride{static_cast<int>(blockDim.x * gridDim.x)};
	for (int i{thread_idx}; i < n; i += stride)
	{
		// Read data into the window.
		for (int j{0}; j < WindowSize; ++j)
		{
			int const idx{i - WindowSize / 2 + j};
			window[j] = (idx < 0 || idx >= n) ? 0 : input[idx];
		}
		// Compute the mean from the window.
		float sum{0};
		for (int j{0}; j < WindowSize; ++j)
		{
			sum += window[j];
		}
		float const mean{sum / WindowSize};
		// Write the mean to the output.
		output[i] = mean;
	}
}

	template <int WindowSize>
__global__ void running_mean_local_memory_array(float const* input,
		float* output, int n)
{
	float window[WindowSize];
	int const thread_idx{
		static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
	int const stride{static_cast<int>(blockDim.x * gridDim.x)};
	for (int i{thread_idx}; i < n; i += stride)
	{
		// Read data into the window.
		for (int j{0}; j < WindowSize; ++j)
		{
			int const idx{i - WindowSize / 2 + j};
			window[j] = (idx < 0 || idx >= n) ? 0 : input[idx];
		}
		// Compute the mean from the window.
		float sum{0};
		for (int j{0}; j < WindowSize; ++j)
		{
			// This index accessing the window array cannot be resolved at the
			// compile time by the compiler, even if such indexing would not
			// affect the correctness of the kernel. The consequence is the
			// compiler will place the window array in the local memory rather
			// than in the register file.
			int const idx{(j + n) % WindowSize};
			sum += window[idx];
		}
		float const mean{sum / WindowSize};
		// Write the mean to the output.
		output[i] = mean;
	}
}

	template <int WindowSize>
cudaError_t launch_running_mean_register_array(float const* input,
		float* output, int n,
		cudaStream_t stream)
{
	dim3 const block_size{256, 1, 1};
	dim3 const grid_size{(n + block_size.x - 1) / block_size.x, 1, 1};
	running_mean_register_array<WindowSize>
		<<<grid_size, block_size, 0, stream>>>(input, output, n);
	return cudaGetLastError();
}

	template <int WindowSize>
cudaError_t launch_running_mean_local_memory_array(float const* input,
		float* output, int n,
		cudaStream_t stream)
{
	dim3 const block_size{256, 1, 1};
	dim3 const grid_size{(n + block_size.x - 1) / block_size.x, 1, 1};
	running_mean_local_memory_array<WindowSize>
		<<<grid_size, block_size, 0, stream>>>(input, output, n);
	return cudaGetLastError();
}

// Verify the correctness of the kernel given a window size and a launch
// function.
	template <int WindowSize>
void verify_running_mean(int n, cudaError_t (*launch_func)(float const*, float*,
			int, cudaStream_t))
{
	std::vector<float> h_input_vec(n, 0.f);
	std::vector<float> h_output_vec(n, 1.f);
	std::vector<float> h_output_vec_ref(n, 2.f);
	// Fill the input vector with values.
	for (int i{0}; i < n; ++i)
	{
		h_input_vec[i] = static_cast<float>(i);
	}
	// Compute the reference output vector.
	for (int i{0}; i < n; ++i)
	{
		float sum{0};
		for (int j{0}; j < WindowSize; ++j)
		{
			int const idx{i - WindowSize / 2 + j};
			float const val{(idx < 0 || idx >= n) ? 0 : h_input_vec[idx]};
			sum += val;
		}
		h_output_vec_ref[i] = sum / WindowSize;
	}
	// Allocate device memory.
	float* d_input;
	float* d_output;
	CHECK_CUDA_ERROR(cudaMalloc(&d_input, n * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_output, n * sizeof(float)));
	// Copy data to the device.
	CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input_vec.data(), n * sizeof(float),
				cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_output, h_output_vec.data(),
				n * sizeof(float), cudaMemcpyHostToDevice));
	// Launch the kernel.
	cudaStream_t stream;
	CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
	CHECK_CUDA_ERROR(launch_func(d_input, d_output, n, stream));
	CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
	// Copy the result back to the host.
	CHECK_CUDA_ERROR(cudaMemcpy(h_output_vec.data(), d_output,
				n * sizeof(float), cudaMemcpyDeviceToHost));
	// Check the result.
	for (int i{0}; i < n; ++i)
	{
		if (h_output_vec.at(i) != h_output_vec_ref.at(i))
		{
			std::cerr << "Mismatch at index " << i << ": " << h_output_vec.at(i)
				<< " != " << h_output_vec_ref.at(i) << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}
	// Free device memory.
	CHECK_CUDA_ERROR(cudaFree(d_input));
	CHECK_CUDA_ERROR(cudaFree(d_output));
	CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

int main()
{
	// Try different window sizes from small to large.
	constexpr int WindowSize{32};
	int const n{8192};
	verify_running_mean<WindowSize>(
			n, launch_running_mean_register_array<WindowSize>);
	verify_running_mean<WindowSize>(
			n, launch_running_mean_local_memory_array<WindowSize>);
	return 0;
}
