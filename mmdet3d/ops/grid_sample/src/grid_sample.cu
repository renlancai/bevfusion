#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 双线性插值核函数
template<typename scalar_t>
__global__ void grid_sample_bilinear_kernel(
    const scalar_t* input,   // 输入图像 [N, C, H_in, W_in]
    const scalar_t* grid,    // 网格坐标 [N, H_out, W_out, 2]
    scalar_t* output,        // 输出图像 [N, C, H_out, W_out]
    int N, int C, int H_in, int W_in,
    int H_out, int W_out) 
{
    // 计算当前线程处理的输出位置
    const int n = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.z % C; // 通道维度展开到blockIdx.z

    if (n >= N || c >= C || h >= H_out || w >= W_out) return;

    // 获取当前网格坐标 (x, y)
    const scalar_t gx = grid[((n * H_out + h) * W_out + w) * 2 + 0];
    const scalar_t gy = grid[((n * H_out + h) * W_out + w) * 2 + 1];

    // 将归一化坐标转换为像素坐标 (align_corners=False)
    scalar_t x = (gx + 1) * (W_in - 1) / 2; // [-1,1] → [0, W_in-1]
    scalar_t y = (gy + 1) * (H_in - 1) / 2;

    // 计算四个相邻点坐标
    int x0 = floor(x);
    int y0 = floor(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // 计算权重
    scalar_t wx = x - x0;
    scalar_t wy = y - y0;

    // 边界处理 (padding_mode='zeros')
    bool x0_valid = (x0 >= 0) && (x0 < W_in);
    bool y0_valid = (y0 >= 0) && (y0 < H_in);
    bool x1_valid = (x1 >= 0) && (x1 < W_in);
    bool y1_valid = (y1 >= 0) && (y1 < H_in);

    // 输入图像索引计算
    const scalar_t* input_ptr = input + n * C * H_in * W_in + c * H_in * W_in;

    // 计算四个点的值（越界则取0）
    scalar_t v00 = (x0_valid && y0_valid) ? input_ptr[y0 * W_in + x0] : 0;
    scalar_t v01 = (x0_valid && y1_valid) ? input_ptr[y1 * W_in + x0] : 0;
    scalar_t v10 = (x1_valid && y0_valid) ? input_ptr[y0 * W_in + x1] : 0;
    scalar_t v11 = (x1_valid && y1_valid) ? input_ptr[y1 * W_in + x1] : 0;

    // 双线性插值
    scalar_t out_val = (1 - wx) * (1 - wy) * v00 +
                      (1 - wx) *    wy  * v01 +
                      wx  * (1 - wy) * v10 +
                      wx  *    wy  * v11;

    // 写入输出
    output[((n * C + c) * H_out + h) * W_out + w] = out_val;
}

// PyTorch封装函数
torch::Tensor grid_sample_bilinear(
    const torch::Tensor input, 
    const torch::Tensor grid) 
{
    // 获取张量维度
    int N = input.size(0);
    int C = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int H_out = grid.size(1);
    int W_out = grid.size(2);

    // 创建输出张量 [N, C, H_out, W_out]
    auto output = torch::zeros({N, C, H_out, W_out}, input.options());

    // 线程块配置（每个block处理16x16像素）
    dim3 block(16, 16);
    dim3 grid((W_out + block.x - 1) / block.x,
              (H_out + block.y - 1) / block.y,
              N * C); // 将通道维度展开

    // 启动核函数
    AT_DISPATCH_FLOATING_TYPES(input.type(), "grid_sample_bilinear", [&] {
        grid_sample_bilinear_kernel<scalar_t><<<grid, block>>>(
            input.data<scalar_t>(),
            grid.data<scalar_t>(),
            output.data<scalar_t>(),
            N, C, H_in, W_in,
            H_out, W_out
        );
    });

    return output;
}

// PyTorch绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grid_sample_bilinear", &grid_sample_bilinear, "Bilinear grid sampling");
}

