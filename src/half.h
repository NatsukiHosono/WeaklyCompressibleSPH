#include <cuda_fp16.h>
#include <cuda_fp16.hpp>

__device__ half sqrt(const half x){
	return hsqrt(x);
}

template <typename real> __device__ bool operator==(const half lhs, const real rhs){
	return (lhs == static_cast<half>(rhs)) ? true : false;
}

template <typename real> __device__ bool operator<(const half lhs, const real rhs){
	return (lhs < static_cast<half>(rhs)) ? true : false;
}

template <typename real> __device__ bool operator>(const half lhs, const real rhs){
	return (lhs > static_cast<half>(rhs)) ? true : false;
}

__device__ half max(const half lhs, const half rhs){
	return (lhs < rhs) ? rhs : lhs;
}


