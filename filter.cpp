/// Copyright (C) 2024 Advanced Micro Devices, Inc
////
//// SPDX-License-Identifier: MIT

#include <iostream>
using namespace std;
#include <math.h>
#include "filter2d.h"
#include "hls_stream.h"
#include "hls_print.h"
#include <cstdio> // Thêm thư viện để sử dụng printf
void memcpy_hls(DTYPE_IN *dest, const DTYPE_IN *src, int size) {
	LOOP_MEMCPY:
	for (int i = 0; i < size; i++) {
//#pragma HLS PIPELINE II=1
        dest[i] = src[i];
    }
}
void Conv2D_0_with_MaxPool(
    DTYPE_IN *bias_0,
    DTYPE_IN *image_in,   // Ảnh đầu vào từ DDR
    DTYPE_IN *coeffs,     // Kernel từ DDR
    DTYPE_OUT *output     // Đầu ra sau MaxPooling lưu vào DDR
)
{
    const int width = 128;
    const int conv_out = 126;         // Kích thước sau Conv2D
    const int pool_out = conv_out / 2; // Kích thước sau MaxPooling: 63
    const unsigned num_filters = 8;

    for (int f = 0; f < num_filters; f++) {
        for (int i = 0; i < conv_out; i += 2) {
            for (int j = 0; j < conv_out; j += 2) {
                #pragma HLS PIPELINE II=1

                DTYPE_OUT pool_max = 0;
                bool first = true;

                // Duyệt qua cửa sổ 2x2 để pooling
                for (int pi = 0; pi < 2; pi++) {
//#pragma HLS UNROLL
                    for (int pj = 0; pj < 2; pj++) {
//#pragma HLS UNROLL
                        int conv_i = i + pi;
                        int conv_j = j + pj;

                        DTYPE_OUT sum = 0;

                        // Tích chập 3x3
                        for (int ki = 0; ki < 3; ki++) {
//#pragma HLS UNROLL
                            for (int kj = 0; kj < 3; kj++) {
//#pragma HLS UNROLL
                                int img_i = conv_i + ki;
                                int img_j = conv_j + kj;
                                DTYPE_IN pixel_val = image_in[img_i * width + img_j];
                                DTYPE_IN weight = coeffs[f * 9 + ki * 3 + kj];

                                DTYPE_OUT temp = pixel_val * weight;
                                sum += temp;
                            }
                        }

                        sum += bias_0[f];
                        if (sum < 0) sum = 0;

                        // Cập nhật max cho pooling
                        if (first || sum > pool_max) {
                            pool_max = sum;
                            first = false;
                        }
                    }
                }

                // Lưu giá trị MaxPooling sau khi duyệt 2x2
                int out_idx = f * (pool_out * pool_out) + (i / 2) * pool_out + (j / 2);
//#pragma HLS PIPELINE II=1
                output[out_idx] = pool_max;
            }
        }
    }
}

void Conv2D_1_with_MaxPool(
    DTYPE_IN *bias_1,
    DTYPE_IN *image_in,   // Đầu vào: 63x63x8
    DTYPE_IN *coeffs,     // Kernel: 16x8x3x3
    DTYPE_OUT *output     // Đầu ra: 30x30x16 (sau MaxPooling)
)
{
    const int in_width = 63, in_height = 63, in_channels = 8;
    const int out_channels = 16;
    const int conv_out = 61;          // 63 - 3 + 1
    const int pool_out = 30; // 30 (vì 61x61 -> 30x30 sau pooling)
    const int kernel_size = 3;

    for (int f = 0; f < out_channels; f++) {
        for (int i = 0; i < conv_out; i += 2) {
            for (int j = 0; j < conv_out; j += 2) {
//                #pragma HLS PIPELINE II=1

                DTYPE_OUT max_val = 0;
                bool first = true;

                // Lặp qua cửa sổ 2x2 để pooling
                for (int pi = 0; pi < 2; pi++) {
                    for (int pj = 0; pj < 2; pj++) {
                        int conv_i = i + pi;
                        int conv_j = j + pj;

                        if (conv_i >= conv_out || conv_j >= conv_out) continue;

                        DTYPE_OUT sum = 0;

                        // Tích chập qua các kênh
                        for (int c = 0; c < in_channels; c++) {
                            for (int ki = 0; ki < kernel_size; ki++) {
                                for (int kj = 0; kj < kernel_size; kj++) {
#pragma HLS PIPELINE II=1
                                    int img_i = conv_i + ki;
                                    int img_j = conv_j + kj;

                                    int img_idx = c * (in_width * in_height) + img_i * in_width + img_j;
                                    int coeff_idx = f * (in_channels * kernel_size * kernel_size) +
                                                    c * (kernel_size * kernel_size) + ki * kernel_size + kj;
                                    DTYPE_IN pixel_val = image_in[img_idx];
                                    DTYPE_IN weight = coeffs[coeff_idx];

                                    sum += pixel_val * weight;
                                }
                            }
                        }

                        sum += bias_1[f];
                        if (sum < 0) sum = 0;

                        // Lấy max cho pooling
                        if (first || sum > max_val) {
                            max_val = sum;
                            first = false;
                        }
                    }
                }

                // Ghi ra DDR sau khi pooling
                int out_idx = f * (pool_out * pool_out) + (i / 2) * pool_out + (j / 2);
                output[out_idx] = max_val;
            }
        }
    }
}

void Conv2D_2_with_MaxPool(
    DTYPE_IN *bias_2,
    DTYPE_IN *image_in,   // Đầu vào: 30x30x16
    DTYPE_IN *coeffs,     // Bộ lọc: 32x16x3x3
    DTYPE_OUT *output     // Đầu ra: 14x14x32
)
{
    const int in_width = 30, in_height = 30;
    const int in_channels = 16;
    const int out_channels = 32;

    const int conv_out_w = 28, conv_out_h = 28;     // Sau Conv2D
    const int pool_out_w = conv_out_w / 2;          // 14
    const int pool_out_h = conv_out_h / 2;          // 14

    for (int f = 0; f < out_channels; f++) {
        for (int i = 0; i < conv_out_h; i += 2) {
            for (int j = 0; j < conv_out_w; j += 2) {
//#pragma HLS PIPELINE II=1

                DTYPE_OUT pool_max = 0;
                bool first = true;

                // Duyệt qua cửa sổ 2x2 cho MaxPooling
                for (int pi = 0; pi < 2; pi++) {
                    for (int pj = 0; pj < 2; pj++) {
                        int conv_i = i + pi;
                        int conv_j = j + pj;

                        DTYPE_OUT sum = 0;

                        // Tích chập 3x3 trên tất cả các kênh đầu vào
                        for (int c = 0; c < in_channels; c++) {
                            for (int ki = 0; ki < 3; ki++) {
                                for (int kj = 0; kj < 3; kj++) {
#pragma HLS PIPELINE II=1
                                    int img_row = conv_i + ki;
                                    int img_col = conv_j + kj;
                                    int img_idx = c * (in_width * in_height) + img_row * in_width + img_col;
                                    int coeff_idx = f * (in_channels * 3 * 3) + c * 9 + ki * 3 + kj;
//#pragma HLS PIPELINE II=1
                                    sum += image_in[img_idx] * coeffs[coeff_idx];
                                }
                            }
                        }

                        // Cộng bias + ReLU
                        sum += bias_2[f];
                        if (sum < 0) sum = 0;

                        // Cập nhật max trong 2x2
                        if (first || sum > pool_max) {
                            pool_max = sum;
                            first = false;
                        }
                    }
                }

                // Lưu kết quả MaxPooling vào DDR
                int out_idx = f * (pool_out_w * pool_out_h) + (i / 2) * pool_out_w + (j / 2);
                output[out_idx] = pool_max;
            }
        }
    }
}


void Conv2D_3_with_MaxPool(
    DTYPE_IN *bias_3,
    DTYPE_IN *image_in,   // 30x30x16
    DTYPE_IN *coeffs,     // 64x32x3x3
    DTYPE_OUT *output     // 14x14x64 (đã đổi thứ tự thành pixel-major)
)
{
    const int in_width = 14, in_height = 14;
    const int in_channels = 32;
    const int out_channels = 64;

    const int conv_out_w = 12, conv_out_h = 12;
    const int pool_out_w = conv_out_w / 2;  // 6
    const int pool_out_h = conv_out_h / 2;  // 6

    for (int i = 0; i < conv_out_h; i += 2) {
        for (int j = 0; j < conv_out_w; j += 2) {
            for (int f = 0; f < out_channels; f++) {
//#pragma HLS PIPELINE II=1

                DTYPE_OUT pool_max = 0;
                bool first = true;

                for (int pi = 0; pi < 2; pi++) {
                    for (int pj = 0; pj < 2; pj++) {
                        int conv_i = i + pi;
                        int conv_j = j + pj;

                        DTYPE_OUT sum = 0;

                        for (int c = 0; c < in_channels; c++) {
                            for (int ki = 0; ki < 3; ki++) {
                                for (int kj = 0; kj < 3; kj++) {
                                    int img_row = conv_i + ki;
                                    int img_col = conv_j + kj;

                                    int img_idx = c * (in_width * in_height) + img_row * in_width + img_col;
                                    int coeff_idx = f * (in_channels * 3 * 3) + c * 9 + ki * 3 + kj;

                                    sum += image_in[img_idx] * coeffs[coeff_idx];
                                }
                            }
                        }

                        sum += bias_3[f];
                        if (sum < 0) sum = 0;

                        if (first || sum > pool_max) {
                            pool_max = sum;
                            first = false;
                        }
                    }
                }

                // Thay đổi vị trí ghi output: pixel-major (p0k0, p0k1, ..., p1k0, ...)
                int pixel_idx = (i / 2) * pool_out_w + (j / 2);
                int out_idx = pixel_idx * out_channels + f;
                output[out_idx] = pool_max;
            }
        }
    }
}


void FullyConnected(
    DTYPE_IN *input,
    DTYPE_IN *kernel_4,   // 230400 phần tử = 100 * 2304
    DTYPE_IN *biases,     // 100 biases
    DTYPE_OUT *output     // 100 output values
) {
#pragma HLS INLINE off

    DTYPE_IN kernel_buffer4[2304];
#pragma HLS ARRAY_PARTITION variable=kernel_buffer4 cyclic factor=4 dim=1


    for (int k = 0; k < 100; k++) {
//#pragma HLS PIPELINE II=1

        // Copy weights cho neuron k
        for (int j = 0; j < 2304; j++) {
//#pragma HLS UNROLL factor=3
            kernel_buffer4[j] = kernel_4[k * 2304 + j];
        }

        DTYPE_OUT acc = biases[k];
        for (int j = 0; j < 2304; j++) {
//#pragma HLS UNROLL factor=3
            acc += input[j] * kernel_buffer4[j];
        }

        // ReLU
        if (acc < 0) acc = 0;

        output[k] = acc;

        // In kết quả output[k] (Debug)
//        printf("output[%d] = %.13f\n", k, (float)output[k]);
    }
}

inline DTYPE_OUT sigmoid(DTYPE_OUT x) {
    DTYPE_OUT y;

    if (x >= DTYPE_OUT(4.0)) {
        y = DTYPE_OUT(1.0);
    } else if (x >= DTYPE_OUT(0.0)) {
        y = DTYPE_OUT(-0.03577) * x * x
            + DTYPE_OUT(0.25908) * x
            + DTYPE_OUT(0.5038);
    } else {
        // Nếu x < 0 → xử lý ngoài theo yêu cầu của bạn (nếu cần)
        y = DTYPE_OUT(0.0); // hoặc dùng sigmoid(-x) nếu muốn đối xứng
    }

    return y;
}


//inline DTYPE_OUT hardsigmoid(DTYPE_OUT x) {
//    DTYPE_OUT y = DTYPE_OUT(0.2) * x + DTYPE_OUT(0.5);
//
//    // Giới hạn y trong khoảng [0, 1]
//    if (y > DTYPE_OUT(1.0)) y = DTYPE_OUT(1.0);
//    else if (y < DTYPE_OUT(0.0)) y = DTYPE_OUT(0.0);
//
//    return y;
//}

// Fully connected layer: 100 input → 1 output
void FullyConnected_2(
    DTYPE_IN *input,       // [100] input vector
    DTYPE_IN *weights,     // [100] weights
    DTYPE_IN *bias,        // 1 bias
    DTYPE_OUT *output      // 1 output
) {
#pragma HLS INLINE off

    DTYPE_OUT acc = *bias;

    // In giá trị bias
    printf("Bias = %.14f\n", (float)*bias);

    // Tính toán acc
    for (int i = 0; i < 100; i++) {
        DTYPE_OUT product = input[i] * weights[i];  // Sản phẩm input[i] * weights[i]
        acc += product;
        printf("input[%d] * weight[%d] = %.14f * %.14f = %.14f\n", i, i, (float)input[i], (float)weights[i], (float)product);
    }

    // In tổng acc sau khi cộng dồn tất cả các phần tử
//    printf("Tổng acc = %.14f\n", (float)acc);

    // Tính output bằng hàm sigmoid
    *output = sigmoid(acc);

    // In giá trị output sau khi tính sigmoid
//    printf("output = %.14f\n", (float)*output);
}


extern "C" {


void CNN(
        DTYPE_IN kernel[72],
		DTYPE_IN kernel_1[1152],
		DTYPE_IN kernel_2[4608],
		DTYPE_IN kernel_3[18432],
		DTYPE_IN kernel_4[230400],
		DTYPE_IN kernel_5[100],
	    DTYPE_IN bias[8],
	    DTYPE_IN bias_1[16],
		DTYPE_IN bias_2[32],
		DTYPE_IN bias_3[64],
		DTYPE_IN bias_4[100],
		DTYPE_IN bias_5[1],
        DTYPE_IN src[MAX_IMAGE_WIDTH*MAX_IMAGE_HEIGHT],
        DTYPE_OUT dst[1]
)
{
#pragma HLS INTERFACE m_axi port=src depth=16384 offset=slave
#pragma HLS INTERFACE m_axi port=dst depth=2304 offset=slave
#pragma HLS INTERFACE m_axi port=bias depth=8 offset=slave
#pragma HLS INTERFACE m_axi port=kernel depth=72 offset=slave
#pragma HLS INTERFACE m_axi port=bias_1 depth=16 offset=slave
#pragma HLS INTERFACE m_axi port=kernel_1 depth=1152 offset=slave
#pragma HLS INTERFACE m_axi port=bias_2 depth=32 offset=slave
#pragma HLS INTERFACE m_axi port=kernel_2 depth=4608 offset=slave
#pragma HLS INTERFACE m_axi port=bias_3 depth=64 offset=slave
#pragma HLS INTERFACE m_axi port=kernel_3 depth=18432 offset=slave
#pragma HLS INTERFACE m_axi port=bias_4 depth=100 offset=slave
#pragma HLS INTERFACE m_axi port=kernel_4 depth=230400 offset=slave
#pragma HLS INTERFACE m_axi port=bias_5 depth=1 offset=slave
#pragma HLS INTERFACE m_axi port=kernel_5 depth=100 offset=slave
#pragma HLS INTERFACE s_axilite port=return
//#pragma HLS DATAFLOW // Kích hoạt chế độ Dataflow
	DTYPE_IN input_buffer [128*128];
	DTYPE_IN kernel_buffer [72];
	DTYPE_IN bias_buffer [8];
	DTYPE_IN kernel_buffer1 [1152];
	DTYPE_IN bias_buffer1 [16];
	DTYPE_IN kernel_buffer2 [4608];
	DTYPE_IN bias_buffer2 [32];
	DTYPE_IN kernel_buffer3 [18432];
    DTYPE_IN bias_buffer3 [64];
    DTYPE_IN kernel_buffer4[2304];  // Đặc biệt cho kernel_4
    DTYPE_IN bias_buffer4[100];
    DTYPE_IN kernel_buffer5[100];  // Đặc biệt cho kernel_4
    DTYPE_IN bias_buffer5[1];
	DTYPE_IN output_buffer [61*61*16];//trung gian
	DTYPE_IN output_buffer1 [63*63*8];// trung gian
//	DTYPE_IN output_buffer2 [61*61*16];// trung gian
	// *** Ánh xạ LUT & FF + PARTITION ***
	// *** Ánh xạ LUT & FF + PARTITION ***
	// *** Ánh xạ LUT & FF + PARTITION ***
	#pragma HLS ARRAY_PARTITION variable=kernel_buffer complete
	#pragma HLS bind_storage variable=kernel_buffer type=ram_1p impl=register

	#pragma HLS ARRAY_PARTITION variable=bias_buffer complete
	#pragma HLS bind_storage variable=bias_buffer type=ram_1p impl=register

	#pragma HLS ARRAY_PARTITION variable=kernel_buffer1 cyclic factor=16
	#pragma HLS bind_storage variable=kernel_buffer1 type=ram_1p impl=register

	#pragma HLS ARRAY_PARTITION variable=bias_buffer1 complete
	#pragma HLS bind_storage variable=bias_buffer1 type=ram_1p impl=register

	#pragma HLS ARRAY_PARTITION variable=kernel_buffer2 cyclic factor=16
	#pragma HLS bind_storage variable=kernel_buffer2 type=ram_1p impl=register

	#pragma HLS ARRAY_PARTITION variable=bias_buffer2 complete
	#pragma HLS bind_storage variable=bias_buffer2 type=ram_1p impl=register

	#pragma HLS ARRAY_PARTITION variable=kernel_buffer3 cyclic factor=16
	#pragma HLS bind_storage variable=kernel_buffer3 type=ram_1p impl=register

	#pragma HLS ARRAY_PARTITION variable=bias_buffer3 complete
	#pragma HLS bind_storage variable=bias_buffer3 type=ram_1p impl=register

  #pragma HLS ARRAY_PARTITION variable=kernel_buffer5 complete
	#pragma HLS bind_storage variable=bias_buffer5 type=ram_1p impl=register

     #pragma HLS ARRAY_PARTITION variable=bias_buffer5 complete
	#pragma HLS bind_storage variable=bias_buffer5 type=ram_1p impl=register

	#pragma HLS ARRAY_PARTITION variable=output_buffer cyclic factor=16
//#pragma HLS bind_storage variable=output_buffer type=ram_1p impl=register
	#pragma HLS ARRAY_PARTITION variable=output_buffer1 cyclic factor=8
//#pragma HLS bind_storage variable=output_buffer1 type=ram_1p impl=register


	memcpy_hls(input_buffer, (const DTYPE_IN *)src, 128*128);
	memcpy_hls(kernel_buffer, (const DTYPE_IN *)kernel, 72 );
	memcpy_hls(bias_buffer, (const DTYPE_IN *)bias, 8 );
	memcpy_hls(kernel_buffer1, (const DTYPE_IN *)kernel_1, 1152 );
    memcpy_hls(bias_buffer1, (const DTYPE_IN *)bias_1, 16 );
    memcpy_hls(kernel_buffer2, (const DTYPE_IN *)kernel_2, 4608 );
    memcpy_hls(bias_buffer2, (const DTYPE_IN *)bias_2, 32 );
    memcpy_hls(kernel_buffer3, (const DTYPE_IN *)kernel_3, 18432 );
    memcpy_hls(bias_buffer3, (const DTYPE_IN *)bias_3, 64 );
    memcpy_hls(bias_buffer4, (const DTYPE_IN *)bias_4, 100 );
    memcpy_hls(kernel_buffer5, (const DTYPE_IN *)kernel_5, 100 );
    memcpy_hls(bias_buffer5, (const DTYPE_IN *)bias_5, 1 );

     Conv2D_0_with_MaxPool(bias_buffer, input_buffer, kernel_buffer, output_buffer1);
     Conv2D_1_with_MaxPool(bias_buffer1, output_buffer1, kernel_buffer1, output_buffer);
     Conv2D_2_with_MaxPool(bias_buffer2, output_buffer, kernel_buffer2, output_buffer1);
     Conv2D_3_with_MaxPool(bias_buffer3, output_buffer1, kernel_buffer3, output_buffer);
     FullyConnected(output_buffer, kernel_4, bias_buffer4, output_buffer1);
     FullyConnected_2(output_buffer1, kernel_buffer5, bias_buffer5, output_buffer);
     memcpy_hls(dst, output_buffer, 1);
   //  printf("output = %.14f\n", (float)*dst);

}
}
