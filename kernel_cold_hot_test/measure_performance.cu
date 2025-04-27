#include <functional>
#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>

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

	template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
		cudaStream_t stream, size_t num_repeats = 100,
		size_t num_warmups = 100, bool flush_l2_cache = false)
{
	int device_id{0};
	int l2_cache_size{0};
	CHECK_CUDA_ERROR(cudaGetDevice(&device_id));
	CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&l2_cache_size,
				cudaDevAttrL2CacheSize, device_id));

	void* l2_flush_buffer{nullptr};
	CHECK_CUDA_ERROR(
			cudaMalloc(&l2_flush_buffer, static_cast<size_t>(l2_cache_size)));

	cudaEvent_t start, stop;
	float time{0.0f};
	float call_time{0.0f};

	CHECK_CUDA_ERROR(cudaEventCreate(&start));
	CHECK_CUDA_ERROR(cudaEventCreate(&stop));

	/*warmup l2 cache, kernel replay*/
	for (size_t i{0}; i < num_warmups; ++i)
	{
		bound_function(stream);
	}

	CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

	for (size_t i{0}; i < num_repeats; ++i)
	{
		if (flush_l2_cache)
		{
			CHECK_CUDA_ERROR(cudaMemsetAsync(l2_flush_buffer, 0,
						static_cast<size_t>(l2_cache_size),
						stream));
			CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
		}
		CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
		/*kernel launch*/
		CHECK_CUDA_ERROR(bound_function(stream));
		CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
		CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&call_time, start, stop));
		time += call_time;
	}
	CHECK_CUDA_ERROR(cudaEventDestroy(start));
	CHECK_CUDA_ERROR(cudaEventDestroy(stop));

	CHECK_CUDA_ERROR(cudaFree(l2_flush_buffer));

	float const latency{time / num_repeats};

	return latency;
}

__global__ void copy(float* output, float const* input, size_t n)
{
	size_t const idx{blockDim.x * blockIdx.x + threadIdx.x};
	size_t const stride{blockDim.x * gridDim.x};
	for (size_t i{idx}; i < n; i += stride)
	{
		output[i] = input[i];
	}
}

cudaError_t launch_copy(float* output, float const* input, size_t n,
		cudaStream_t stream)
{
	dim3 const threads_per_block{1024};
	dim3 const blocks_per_grid{32};
	copy<<<blocks_per_grid, threads_per_block, 0, stream>>>(output, input, n);
	return cudaGetLastError();
}

int main()
{
	int device_id{0};
	CHECK_CUDA_ERROR(cudaGetDevice(&device_id));
	cudaDeviceProp device_prop;
	CHECK_CUDA_ERROR(cudaGetDeviceProperties(&device_prop, device_id));
	std::cout << "Device Name: " << device_prop.name << std::endl;
	float const memory_size{static_cast<float>(device_prop.totalGlobalMem) /
		(1 << 30)};
	std::cout << "DRAM Size: " << memory_size << " GB" << std::endl;
	float const peak_bandwidth{
		static_cast<float>(2.0f * device_prop.memoryClockRate *
				(device_prop.memoryBusWidth / 8) / 1.0e6)};
	std::cout << "DRAM Peak Bandwitdh: " << peak_bandwidth << " GB/s"
		<< std::endl;
	int const l2_cache_size{device_prop.l2CacheSize};
	float const l2_cache_size_mb{static_cast<float>(l2_cache_size) / (1 << 20)};
	std::cout << "L2 Cache Size: " << l2_cache_size_mb << " MB" << std::endl;

	constexpr size_t num_repeats{10000};
	constexpr size_t num_warmups{1000};

	size_t const n{l2_cache_size / 2 / sizeof(float)};
	cudaStream_t stream;

	float *d_input, *d_output;

	CHECK_CUDA_ERROR(cudaMalloc(&d_input, n * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_output, n * sizeof(float)));

	CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

	std::function<cudaError_t(cudaStream_t)> function{
		std::bind(launch_copy, d_output, d_input, n, std::placeholders::_1)};

	float const hot_latency{
		measure_performance(function, stream, num_repeats, num_warmups, false)};
	std::cout << std::fixed << std::setprecision(4)
		<< "Hot Latency: " << hot_latency << " ms" << std::endl;

	float const cold_latency{
		measure_performance(function, stream, num_repeats, num_warmups, true)};
	std::cout << std::fixed << std::setprecision(4)
		<< "Cold Latency: " << cold_latency << " ms" << std::endl;

	CHECK_CUDA_ERROR(cudaFree(d_input));
	CHECK_CUDA_ERROR(cudaFree(d_output));
	CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}
