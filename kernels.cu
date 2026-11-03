// LayerNorm and (causal, scaled) softmax.
// both use the same pattern, one thread block owns one row. each thread walks the row in steps of BLOCK_SIZE accumulating a partial result, then the block combines the partials in shared memory.
#include <torch/extension.h>

#define BLOCK_SIZE 256 // threads per block: full warps (multiple of 32), and a power of 2 for the reduction

// combine one value per thread into a single sum (or max) that every thread gets back. tree reduction:
// at each step the first half of the active threads fold in the values of the second half, so the
// number of values halves: 256 -> 128 -> ... -> 1. thread i reads shared_values[i] and shared_values[i + stride], so a
// warp touches 32 consecutive floats (no bank conflicts), and the idle threads are always the
// high-numbered ones, so whole warps go idle together (no divergence until fewer than 32 remain).
__device__ float block_reduce(float value, bool is_max) {
    __shared__ float shared_values[BLOCK_SIZE];
    int thread_id = threadIdx.x;
    shared_values[thread_id] = value;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (thread_id < stride) {
            float other_value = shared_values[thread_id + stride];
            if (is_max) {
                shared_values[thread_id] = fmaxf(shared_values[thread_id], other_value);
            } else {
                shared_values[thread_id] = shared_values[thread_id] + other_value;
            }
        }
        __syncthreads(); // outside the if, since every thread in the block must reach it
    }
    float result = shared_values[0];
    __syncthreads(); // all threads must read shared_values[0] before the next call overwrites it
    return result;
}

// output = (input - mean) / sqrt(variance + epsilon) * weight + bias, normalizing over each row of length row_length.
// in every loop, neighboring threads read neighboring floats, so global memory accesses are coalesced
__global__ void layernorm_kernel(const float* input, const float* weight, const float* bias, float* output, int row_length, float epsilon) {
    input += blockIdx.x * row_length; output += blockIdx.x * row_length; // jump to this block's row
    int thread_id = threadIdx.x;

    float sum = 0.0f;
    for (int i = thread_id; i < row_length; i += BLOCK_SIZE) sum += input[i];
    float mean = block_reduce(sum, false) / row_length;

    float squared_difference_sum = 0.0f;
    for (int i = thread_id; i < row_length; i += BLOCK_SIZE) {
        float difference = input[i] - mean; squared_difference_sum += difference * difference;
    }
    float inverse_standard_deviation = rsqrtf(block_reduce(squared_difference_sum, false) / row_length + epsilon);

    for (int i = thread_id; i < row_length; i += BLOCK_SIZE) {
        output[i] = (input[i] - mean) * inverse_standard_deviation * weight[i] + bias[i];
    }
}

// attention softmax with the scale and causal mask fused in. input is (batch_size, num_heads, num_queries, num_keys):
// num_queries new queries attending over num_keys keys, the first num_keys - num_queries of which come from the KV cache.
// query t sits at absolute position (num_keys - num_queries) + t, so it may only see keys 0..(num_keys - num_queries) + t.
__global__ void softmax_kernel(const float* input, float* output, int num_queries, int num_keys, float scale) {
    input += blockIdx.x * num_keys; output += blockIdx.x * num_keys; // jump to this block's row
    int thread_id = threadIdx.x;
    int query_index = blockIdx.x % num_queries;
    int num_visible_keys = (num_keys - num_queries) + query_index + 1; // one past the last key this query is allowed to see

    float max_value = -INFINITY; // subtract the max before exp() for numerical stability
    for (int i = thread_id; i < num_visible_keys; i += BLOCK_SIZE) {
        max_value = fmaxf(max_value, input[i]);
    }
    max_value = block_reduce(max_value, true);

    float sum = 0.0f;
    for (int i = thread_id; i < num_visible_keys; i += BLOCK_SIZE) {
        sum += expf((input[i] - max_value) * scale);
    }
    sum = block_reduce(sum, false);

    for (int i = thread_id; i < num_keys; i += BLOCK_SIZE) {
        if (i < num_visible_keys) {
            output[i] = expf((input[i] - max_value) * scale) / sum;
        } else {
            output[i] = 0.0f;
        }
    }
}

// ---- PyTorch bindings: allocate the output, launch one block per row ----

#define CHECK_INPUT(tensor) TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() && tensor.dtype() == torch::kFloat32, #tensor " must be a contiguous float32 CUDA tensor")

torch::Tensor layernorm(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, double epsilon) {
    CHECK_INPUT(input); CHECK_INPUT(weight); CHECK_INPUT(bias);
    auto output = torch::empty_like(input);
    int row_length = input.size(-1), num_rows = input.numel() / row_length;
    layernorm_kernel<<<num_rows, BLOCK_SIZE>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), row_length, epsilon);
    return output;
}

torch::Tensor softmax(torch::Tensor input, double scale) {
    CHECK_INPUT(input);
    auto output = torch::empty_like(input);
    int num_queries = input.size(-2), num_keys = input.size(-1), num_rows = input.numel() / num_keys;
    softmax_kernel<<<num_rows, BLOCK_SIZE>>>(input.data_ptr<float>(), output.data_ptr<float>(), num_queries, num_keys, scale);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("layernorm", &layernorm);
    module.def("softmax", &softmax);
}
