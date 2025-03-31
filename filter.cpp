
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
void WriteToMem(
        hls::stream<DTYPE_OUT> &pixel_stream_out,
        DTYPE_OUT *dst)
{

    // Kích thước sau Conv + Pooling
    int new_width = 6;
    int new_height = 6;
    unsigned num_pixels = new_width * new_height;
    unsigned num_channels = 64; // 16 kênh đầu ra

    // Lưu kết quả theo từng kênh riêng biệt
    for (int c = 0; c < num_channels; c++) {
        hls::print("Channel %d:\n", c ); // In ra thông tin kênh
        for (int n = 0; n < num_pixels; n++) {
    #pragma HLS LOOP_TRIPCOUNT max=36
        	   DTYPE_OUT pix = pixel_stream_out.read();
        	              dst[c * num_pixels + n] = pix;

//        	              // Tách từng giá trị in riêng

        	    }
//        	    hls::print("\n");
        }
}


//    // In lại dst theo thứ tự i * 8 + k
//    hls::print("Printing dst in (i * 8 + k) order:\n");
//    for (int k = 0; k < num_channels; k++) {
//    for (int i = 0; i < num_pixels; i++) {
////        for (int k = 0; k < num_channels; k++) {
//            int index = i * 8 + k;
//
//            hls::print("dst index = ");
//            hls::print("%d\n", index);
//
//            hls::print("dst value = ");
//            hls::print("%.14f\n", float(dst[index]));
//        }
//        hls::print("\n"); // Xuống dòng sau mỗi pixel
//    }
//}
//
void ReadFromMem(
         DTYPE_IN  *coeffs,
		 DTYPE_IN *src,
        hls::stream<DTYPE_IN>   &coeff_stream,
        hls::stream<DTYPE_IN>     &pixel_stream_in)
{

    unsigned num_filters = 8; // Sá»‘ lÆ°á»£ng filter
    unsigned filter_size = 3 * 3; // Má»—i filter lÃ  3x3
    unsigned num_coefs = num_filters * filter_size; // Tá»•ng sá»‘ coefficient
//    unsigned num_coefs_padded = (((num_coefs-1)/64)+1)*64; // Make sure number of reads of multiple of 64, enables auto-widening
    read_coefs: for (int i=0; i<num_coefs; i++) {
       DTYPE_IN coef = coeffs[i];
            coeff_stream.write( coef );
    }
    read_image: for (int n = 0; n < 128*128; n++) {
#pragma HLS LOOP_TRIPCOUNT max=16384
        DTYPE_IN pix = src[n];
        pixel_stream_in.write( pix );

     }
}


void Conv2D_0(
    DTYPE_IN *bias_0,
    hls::stream<DTYPE_IN> &coeff_stream,  // Nhận hệ số kernel từ stream
    hls::stream<DTYPE_IN> &image_stream_in,  // Nhận ảnh từ stream
    hls::stream<DTYPE_OUT> &pixel_stream_out // Kết quả đầu ra
) {

    // Bộ đệm lưu ảnh (1D)
    DTYPE_IN image_buffer[128 * 128];
#pragma HLS ARRAY_PARTITION variable=image_buffer cyclic factor=4 dim=1


    // Bộ đệm lưu kernel (1D)
    DTYPE_IN coeff_buffer[8 * 9];
//    #pragma HLS ARRAY_PARTITION variable=coeff_buffer complete dim=1

    // Load ảnh từ `image_stream_in` vào buffer

    for (int i = 0; i < 128 * 128; i++) {
         #pragma HLS PIPELINE II=1
         image_buffer[i] = image_stream_in.read();
     }

    // Load kernel từ `coeff_stream` vào buffer 1D
    for (int i = 0; i < 8 * 9; i++) {
        #pragma HLS PIPELINE II=1
        coeff_buffer[i] = coeff_stream.read();
    }

    // Tích chập
    for (int f = 0; f < 8; f++) {
        for (int i = 0; i < 126; i++) {
            for (int j = 0; j < 126; j++) {
                #pragma HLS PIPELINE II=1
                DTYPE_OUT sum = 0;

                for (int row = 0; row < 3; row++) {
                    for (int col = 0; col < 3; col++) {
                        int img_idx = (i + row) * 128 + (j + col); // Chuyển đổi chỉ số 2D -> 1D
                        int coeff_idx = f * 9 + (row * 3 + col);  // Chuyển đổi chỉ số kernel 2D -> 1D
                        sum += image_buffer[img_idx] * coeff_buffer[coeff_idx];
                    }
                }

                // Cộng bias và ReLU
                sum += bias_0[f];
                if (sum < 0) sum = 0;

                // Ghi kết quả ra stream
                pixel_stream_out.write(sum);
            }
        }
    }
}



void MaxPooling2D(
//        int width,
//        int height,
        hls::stream<DTYPE_OUT> &pixel_stream_in,
        hls::stream<DTYPE_OUT> &pixel_stream_out)
{
    const unsigned num_channels = 8;
    const int new_width =   126;
    const int new_height =  126;

    // Buffer dòng để lưu tạm giá trị từ luồng dữ liệu
    DTYPE_OUT row_buffer[8][2][126];
//    #pragma HLS ARRAY_PARTITION variable=row_buffer dim=1 complete
//    #pragma HLS ARRAY_PARTITION variable=row_buffer dim=2 complete

    // Duyệt qua từng kênh
    for (int c = 0; c < num_channels; c++) {
        for (int i = 0; i < new_height; i++) {
            for (int j = 0; j <new_width; j++) {
//                #pragma HLS PIPELINE II=1
                row_buffer[c][i % 2][j] = pixel_stream_in.read();

                if (i % 2 == 1 && j % 2 == 1) {
                    DTYPE_OUT p1 = row_buffer[c][0][j-1]; // (i-1, j-1)
                    DTYPE_OUT p2 = row_buffer[c][0][j];   // (i-1, j)
                    DTYPE_OUT p3 = row_buffer[c][1][j-1]; // (i, j-1)
                    DTYPE_OUT p4 = row_buffer[c][1][j];   // (i, j)

                    // Lấy max
                    DTYPE_OUT max_val = (p1 > p2) ? p1 : p2;
                    max_val = (max_val > p3) ? max_val : p3;
                    max_val = (max_val > p4) ? max_val : p4;

                    // Ghi vào stream output
                    pixel_stream_out.write(max_val);
                }
            }
        }
    }
}



void Conv2D_1(
		DTYPE_IN *bias_1,
		DTYPE_IN *coeffs1,
        hls::stream<DTYPE_IN> &pixel_stream_in,
	    hls::stream<DTYPE_OUT> &pixel_stream_out
        )
{
    const int num_filters = 16;
    const int num_channels = 8;

    // Buffer lưu đầu vào kích thước 63x63x8
    DTYPE_IN input_buffer[8][63][63] = {0};
//    #pragma HLS ARRAY_PARTITION variable=input_buffer dim=1 complete

    // Đọc dữ liệu vào buffer
    for (int c = 0; c < num_channels; c++) {
        for (int i = 0; i < 63; i++) {
            for (int j = 0; j < 63; j++) {
//                #pragma HLS PIPELINE II=1
                input_buffer[c][i][j] = pixel_stream_in.read();
            }
        }
    }

    for (int f = 0; f < num_filters; f++) {
        for (int i = 0; i < 61; i++) {
            for (int j = 0; j < 61; j++) {
//                #pragma HLS PIPELINE II=1

                // Khởi tạo sum
                DTYPE_OUT sum = 0;

                // Tích chập 8 kênh
                for (int c = 0; c < num_channels; c++) {
                    for (int row = 0; row < 3; row++) {
                        for (int col = 0; col < 3; col++) {
                            int index = f * (num_channels * 3 * 3) + c * (3 * 3) + row * 3 + col;
                                                DTYPE_OUT temp = input_buffer[c][i + row][j + col] * coeffs1[index];
                                                sum += temp;

                        }
                    }
                }

                // Cộng bias sau khi nhân kernel
                sum += bias_1[f];

                // ReLU activation
                if (sum < 0) sum = 0;

                // Lưu vào buffer
                pixel_stream_out.write(sum);
            }
        }
    }
}


void MaxPooling2D_1(
    hls::stream<DTYPE_OUT> &pixel_stream_in,
    hls::stream<DTYPE_OUT> &pixel_stream_out)
{
    const unsigned num_channels = 16;
    const int input_width = 61, input_height = 61;
    const int output_width = 30, output_height = input_height / 2 + (input_height % 2);

    // Bộ đệm dòng (lưu 2 hàng gần nhất)
    DTYPE_OUT row_buffer[16][2][61];
//    #pragma HLS ARRAY_PARTITION variable=row_buffer dim=1 complete
//    #pragma HLS ARRAY_PARTITION variable=row_buffer dim=2 complete

    // Duyệt từng kênh
    for (int c = 0; c < num_channels; c++) {
        for (int i = 0; i < input_height; i++) {
            for (int j = 0; j < input_width; j++) {
//                #pragma HLS PIPELINE II=1
                row_buffer[c][i % 2][j] = pixel_stream_in.read();

                // Đảm bảo không lấy hàng cuối và cột cuối
                if (i % 2 == 1 && j % 2 == 1 && i < input_height - 1 && j < input_width - 1) {
                    DTYPE_OUT p1 = row_buffer[c][0][j-1]; // (i-1, j-1)
                    DTYPE_OUT p2 = row_buffer[c][0][j];   // (i-1, j)
                    DTYPE_OUT p3 = row_buffer[c][1][j-1]; // (i, j-1)
                    DTYPE_OUT p4 = row_buffer[c][1][j];   // (i, j)


                    // Lấy max
                    DTYPE_OUT max_val = (p1 > p2) ? p1 : p2;
                    max_val = (max_val > p3) ? max_val : p3;
                    max_val = (max_val > p4) ? max_val : p4;
//                    printf("Channel %d, i=%d, j=%d: p1=%.14f, p2=%.14f, p3=%.14f, p4=%.14f, max=%.14f\n",
//                                                              c, i, j, (float)p1, (float)p2, (float)p3, (float)p4, (float)max_val);
                                    // Lấy max
                    // Ghi vào output
                    pixel_stream_out.write(max_val);
                }
            }
        }
    }
}


void Conv2D_2(
        DTYPE_IN *bias_2,
        DTYPE_IN *coeffs2,
        hls::stream<DTYPE_IN> &pixel_stream_in,
        hls::stream<DTYPE_OUT> &pixel_stream_out
        )
{
    const int num_filters = 32;
    const int num_channels = 16;

    // Buffer lưu đầu vào kích thước 30x30x16
    DTYPE_IN input_buffer[16][30][30] = {0};
//    #pragma HLS ARRAY_PARTITION variable=input_buffer dim=1 complete

    // Đọc dữ liệu vào buffer
    for (int c = 0; c < num_channels; c++) {
        for (int i = 0; i < 30; i++) {
            for (int j = 0; j < 30; j++) {
//                #pragma HLS PIPELINE II=1
                input_buffer[c][i][j] = pixel_stream_in.read();
            }
        }
    }

    for (int f = 0; f < num_filters; f++) {
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
//                #pragma HLS PIPELINE II=1

                // Khởi tạo sum
                DTYPE_OUT sum = 0;

                // Tích chập 16 kênh
                for (int c = 0; c < num_channels; c++) {
                    for (int row = 0; row < 3; row++) {
                        for (int col = 0; col < 3; col++) {
                            int index = f * (num_channels * 3 * 3) + c * (3 * 3) + row * 3 + col;
                            sum += input_buffer[c][i + row][j + col] *  coeffs2[index];
                        }
                    }
                }

                // Cộng bias sau khi nhân kernel
                sum += bias_2[f];

                // ReLU activation
                if (sum < 0) sum = 0;
//                printf("%.14f\n", (float)sum);



                // Lưu vào buffer
                pixel_stream_out.write(sum);
            }
        }
    }
}

void MaxPooling2D_2(
//        int width,
//        int height,
        hls::stream<DTYPE_OUT> &pixel_stream_in,
        hls::stream<DTYPE_OUT> &pixel_stream_out)
{
    const unsigned num_channels = 32;
    const int new_width =   28;
    const int new_height =  28;

    // Buffer dòng để lưu tạm giá trị từ luồng dữ liệu
    DTYPE_OUT row_buffer[32][2][28];
//    #pragma HLS ARRAY_PARTITION variable=row_buffer dim=1 complete
//    #pragma HLS ARRAY_PARTITION variable=row_buffer dim=2 complete

    // Duyệt qua từng kênh
    for (int c = 0; c < num_channels; c++) {
        for (int i = 0; i < new_height; i++) {
            for (int j = 0; j <new_width; j++) {
//                #pragma HLS PIPELINE II=1
                row_buffer[c][i % 2][j] = pixel_stream_in.read();

                if (i % 2 == 1 && j % 2 == 1) {
                    DTYPE_OUT p1 = row_buffer[c][0][j-1]; // (i-1, j-1)
                    DTYPE_OUT p2 = row_buffer[c][0][j];   // (i-1, j)
                    DTYPE_OUT p3 = row_buffer[c][1][j-1]; // (i, j-1)
                    DTYPE_OUT p4 = row_buffer[c][1][j];   // (i, j)

                    // Lấy max
                    DTYPE_OUT max_val = (p1 > p2) ? p1 : p2;
                    max_val = (max_val > p3) ? max_val : p3;
                    max_val = (max_val > p4) ? max_val : p4;

                    // Ghi vào stream output
                    pixel_stream_out.write(max_val);
                }
            }
        }
    }
}
void Conv2D_3(
	    DTYPE_IN *bias_3,
		DTYPE_IN *coeffs3,
	     // Nhận hệ số kernel từ stream
	    hls::stream<DTYPE_IN> &pixel_stream_in,  // Nhận ảnh từ stream
	    hls::stream<DTYPE_OUT> &pixel_stream_out // Kết quả đầu ra
        )
{
    const int num_filters = 64;
    const int num_channels = 32;

    // Buffer lưu đầu vào kích thước 30x30x16
    DTYPE_IN input_buffer[32][14][14] = {0};
//    #pragma HLS ARRAY_PARTITION variable=input_buffer dim=1 complete

    // Đọc dữ liệu vào buffer
    for (int c = 0; c < num_channels; c++) {
        for (int i = 0; i < 14; i++) {
            for (int j = 0; j < 14; j++) {
//                #pragma HLS PIPELINE II=1
                input_buffer[c][i][j] = pixel_stream_in.read();
            }
        }
    }

    for (int f = 0; f < num_filters; f++) {
        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 12; j++) {
//                #pragma HLS PIPELINE II=1

                // Khởi tạo sum
                DTYPE_OUT sum = 0;

                // Tích chập 16 kênh
                for (int c = 0; c < num_channels; c++) {
                    for (int row = 0; row < 3; row++) {
                        for (int col = 0; col < 3; col++) {
                            int index = f * (num_channels * 3 * 3) + c * (3 * 3) + row * 3 + col;
                            sum += input_buffer[c][i + row][j + col] *  coeffs3[index];
                        }
                    }
                }

                // Cộng bias sau khi nhân kernel
                sum += bias_3[f];

                // ReLU activation
                if (sum < 0) sum = 0;
                  printf("%.14f\n", (float)sum);
//                printf("Filter %d, Position (%d, %d): sum = %.14f\n", f, i, j, (float)sum);

                // Lưu vào buffer
                pixel_stream_out.write(sum);
            }
        }
    }
}
void MaxPooling2D_3(
        hls::stream<DTYPE_OUT> &pixel_stream_in,
        hls::stream<DTYPE_OUT> &pixel_stream_out)
{
    const unsigned num_channels = 64;
    const int new_width =   12;
    const int new_height =  12;

    // Buffer dòng để lưu tạm giá trị từ luồng dữ liệu
    DTYPE_OUT row_buffer[64][2][12];
//    #pragma HLS ARRAY_PARTITION variable=row_buffer dim=1 complete
//    #pragma HLS ARRAY_PARTITION variable=row_buffer dim=2 complete

    // Duyệt qua từng kênh
    for (int c = 0; c < num_channels; c++) {
        for (int i = 0; i < new_height; i++) {
            for (int j = 0; j <new_width; j++) {
//                #pragma HLS PIPELINE II=1
                row_buffer[c][i % 2][j] = pixel_stream_in.read();

                if (i % 2 == 1 && j % 2 == 1) {
                    DTYPE_OUT p1 = row_buffer[c][0][j-1]; // (i-1, j-1)
                    DTYPE_OUT p2 = row_buffer[c][0][j];   // (i-1, j)
                    DTYPE_OUT p3 = row_buffer[c][1][j-1]; // (i, j-1)
                    DTYPE_OUT p4 = row_buffer[c][1][j];   // (i, j)

                    // Lấy max
                    DTYPE_OUT max_val = (p1 > p2) ? p1 : p2;
                    max_val = (max_val > p3) ? max_val : p3;
                    max_val = (max_val > p4) ? max_val : p4;
                    printf("p1: %.14f, p2: %.14f, p3: %.14f, p4: %.14f -> max: %.14f\n",
                              (float)p1, (float)p2, (float)p3, (float)p4, (float)max_val);
                    // Ghi vào stream output
                    pixel_stream_out.write(max_val);
                }
            }
        }
    }
}

extern "C" {


void filter2d_accel(
        DTYPE_IN kernel[72],
		DTYPE_IN kernel_1[1152],
		DTYPE_IN kernel_2[4608],
		DTYPE_IN kernel_3[18432],
	    DTYPE_IN bias[8],
	    DTYPE_IN bias_1[16],
		DTYPE_IN bias_2[32],
		DTYPE_IN bias_3[64],
        DTYPE_IN src[MAX_IMAGE_WIDTH*MAX_IMAGE_HEIGHT],
        DTYPE_OUT dst[6*6*64]
)
{
#pragma HLS INTERFACE mode=m_axi bundle=bus_src depth=16384 port=src
#pragma HLS INTERFACE mode=m_axi bundle=bus_dst depth=15876 port=dst
#pragma HLS INTERFACE mode=m_axi bundle=bus_bias depth=8 port=bias
#pragma HLS INTERFACE mode=m_axi bundle=bus_kernel depth=72 port=kernel
#pragma HLS INTERFACE mode=m_axi bundle=bus_bias1 depth=16 port=bias_1
#pragma HLS INTERFACE mode=m_axi bundle=bus_kernel1 depth=1152 port=kernel_1
#pragma HLS INTERFACE mode=m_axi bundle=bus_bias2 depth=32 port=bias_2
#pragma HLS INTERFACE mode=m_axi bundle=bus_kernel2 depth=4608 port=kernel_2
#pragma HLS INTERFACE mode=m_axi bundle=bus_bias3 depth=32 port=bias_3
#pragma HLS INTERFACE mode=m_axi bundle=bus_kernel3 depth=4608 port=kernel_3
#pragma HLS INTERFACE mode=s_axilite port=return bundle=CTRL
//#pragma HLS DATAFLOW
    hls::stream<DTYPE_IN,3>      coefs_stream,coefs_stream2;
    hls::stream<DTYPE_IN,2>      pixel_stream;
    hls::stream<DTYPE_OUT,8>    output_stream, output_stream2,output_stream3,output_stream4;
    hls::stream<DTYPE_OUT,8>    maxpool_stream,maxpool_stream2,maxpool_stream3,maxpool_stream4;
    ReadFromMem(kernel,src,coefs_stream,pixel_stream);
    Conv2D_0(bias,coefs_stream,pixel_stream,output_stream);
    MaxPooling2D (output_stream, maxpool_stream);
    Conv2D_1(bias_1,kernel_1,maxpool_stream,output_stream2);
    MaxPooling2D_1(output_stream2,maxpool_stream2);
    Conv2D_2(bias_2,kernel_2,maxpool_stream2,output_stream3);
    MaxPooling2D_2(output_stream3,maxpool_stream3);
    Conv2D_3(bias_3,kernel_3,maxpool_stream3,output_stream4);
    MaxPooling2D_3(output_stream4,maxpool_stream4);
    WriteToMem(maxpool_stream4, dst);
}
}
