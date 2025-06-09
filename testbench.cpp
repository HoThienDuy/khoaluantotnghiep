// Copyright (C) 2024 Advanced Micro Devices, Inc
//
// SPDX-License-Identifier: MIT

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "filter2d.h"
#include <iostream>

using namespace std;



int main()
{
    DTYPE_IN input[16384];     // 128 x 128 (ảnh đầu vào)
    DTYPE_OUT output[1]; // Kết quả từ lớp Conv2D thứ hai



    // 8 bộ lọc, mỗi bộ lọc 3x3 -> 8 * 9 = 72
    DTYPE_IN kernel1[72];
    DTYPE_IN bias1[8];

    // 16 bộ lọc, mỗi bộ lọc 3x3x8 -> 16 * 8 * 9 = 1152
    DTYPE_IN kernel2[1152];
    DTYPE_IN bias2[16];
    DTYPE_IN kernel3[4608];
    DTYPE_IN bias3[32];
    DTYPE_IN kernel4[18432];
    DTYPE_IN bias4[64];
    DTYPE_IN kernel5[230400];
    DTYPE_IN bias5[100];
    DTYPE_IN kernel6[100];
    DTYPE_IN bias6[1];

    // Mở file chứa input, kernel, bias
    FILE *fp = fopen("C:\\Users\\ADMIN88\\Downloads\\weights_output_cot\\Healthy.txt", "r");
    FILE *fp_kernel1 = fopen("C:\\Users\\ADMIN88\\Downloads\\weights_output_cot\\conv2d_4_weights.txt", "r");
    FILE *fp_bias1 = fopen("C:\\Users\\ADMIN88\\Downloads\\weights_output_cot\\conv2d_4_bias.txt", "r");
    FILE *fp_kernel2 = fopen("C:\\Users\\ADMIN88\\Downloads\\weights_output_cot\\conv2d_5_weights.txt", "r");
    FILE *fp_bias2 = fopen("C:\\Users\\ADMIN88\\Downloads\\weights_output_cot\\conv2d_5_bias.txt", "r");
    FILE *fp_kernel3 = fopen("C:\\Users\\ADMIN88\\Downloads\\weights_output_cot\\conv2d_6_weights.txt", "r");
    FILE *fp_bias3 = fopen("C:\\Users\\ADMIN88\\Downloads\\weights_output_cot\\conv2d_6_bias.txt", "r");
    FILE *fp_kernel4 = fopen("C:\\Users\\ADMIN88\\Downloads\\weights_output_cot\\conv2d_7_weights.txt", "r");
    FILE *fp_bias4 = fopen("C:\\Users\\ADMIN88\\Downloads\\weights_output_cot\\conv2d_7_bias.txt", "r");
    FILE *fp_kernel5 = fopen("C:\\Users\\ADMIN88\\Downloads\\weights_output_cot\\dense_2_weights.txt", "r");
    FILE *fp_bias5 = fopen("C:\\Users\\ADMIN88\\Downloads\\weights_output_cot\\dense_2_bias.txt", "r");
    FILE *fp_kernel6 = fopen("C:\\Users\\ADMIN88\\Downloads\\weights_output_cot\\dense_3_weights.txt", "r");
    FILE *fp_bias6 = fopen("C:\\Users\\ADMIN88\\Downloads\\weights_output_cot\\dense_3_bias.txt", "r");
    FILE *fp_output = fopen("C:\\Users\\ADMIN88\\Downloads\\In_out\\dst2.txt", "w");


    float temp;

    // Đọc ảnh đầu vào từ file
    for (int i = 0; i < 16384; i++) {
        fscanf(fp, "%f", &temp);
        input[i] = (DTYPE_IN) temp;
    }

    // Đọc kernel1 từ file (72 giá trị)
    for (int i = 0; i < 72; i++) {
        fscanf(fp_kernel1, "%f", &temp);
        kernel1[i] = (DTYPE_IN) temp;
    }

    // Đọc bias1 từ file (8 giá trị)
    for (int i = 0; i < 8; i++) {
        fscanf(fp_bias1, "%f", &temp);
        bias1[i] = (DTYPE_IN) temp;
    }

    // Đọc kernel2 từ file (1152 giá trị)
    for (int i = 0; i < 1152; i++) {
        fscanf(fp_kernel2, "%f", &temp);
        kernel2[i] = (DTYPE_IN) temp;
    }

    // Đọc bias2 từ file (16 giá trị)
    for (int i = 0; i < 16; i++) {
        fscanf(fp_bias2, "%f", &temp);
        bias2[i] = (DTYPE_IN) temp;
    }
    // Đọc kernel2 từ file (1152 giá trị)
     for (int i = 0; i < 4608; i++) {
         fscanf(fp_kernel3, "%f", &temp);
         kernel3[i] = (DTYPE_IN) temp;
     }

     // Đọc bias2 từ file (16 giá trị)
     for (int i = 0; i < 32; i++) {
         fscanf(fp_bias3, "%f", &temp);
         bias3[i] = (DTYPE_IN) temp;
     }
     // Đọc kernel2 từ file (1152 giá trị)
        for (int i = 0; i < 18432; i++) {
            fscanf(fp_kernel4, "%f", &temp);
            kernel4[i] = (DTYPE_IN) temp;
        }

        // Đọc bias2 từ file (16 giá trị)
        for (int i = 0; i < 64; i++) {
            fscanf(fp_bias4, "%f", &temp);
            bias4[i] = (DTYPE_IN) temp;
        }
//         Đọc kernel2 từ file (1152 giá trị)
           for (int i = 0; i < 230400; i++) {
               fscanf(fp_kernel5, "%f", &temp);
               kernel5[i] = (DTYPE_IN) temp;
           }

           // Đọc bias2 từ file (16 giá trị)
           for (int i = 0; i < 100; i++) {
               fscanf(fp_bias5, "%f", &temp);
               bias5[i] = (DTYPE_IN) temp;
           }
           for (int i = 0; i < 100; i++) {
                  fscanf(fp_kernel6, "%f", &temp);
                  kernel6[i] = (DTYPE_IN) temp;
              }

              // Đọc bias2 từ file (16 giá trị)
              for (int i = 0; i < 1; i++) {
                  fscanf(fp_bias6, "%f", &temp);
                  bias6[i] = (DTYPE_IN) temp;
              }

       // Đóng fi

    // Đóng file đầu vào sau khi đọc xong
    fclose(fp);
    fclose(fp_kernel1);
    fclose(fp_bias1);
    fclose(fp_kernel2);
    fclose(fp_bias2);
    fclose(fp_kernel3);
    fclose(fp_bias3);
    fclose(fp_kernel4);
    fclose(fp_bias4);
    fclose(fp_kernel5);
    fclose(fp_bias5);
    fclose(fp_kernel6);
    fclose(fp_bias6);
    // Gọi hàm xử lý Conv2D thứ hai
    CNN(kernel1,kernel2,kernel3,kernel4,kernel5,kernel6,bias1,bias2,bias3,bias4,bias5,bias6,input, output);

    // Ghi kết quả từ Conv2D thứ hai vào file
//    for (int k = 0; k < 64; k++) {
//        fprintf(fp_output, "Kênh [%d]:\n", k);

        for (int i = 0; i < 1; i++) { // 61x61 = 3721
            fprintf(fp_output, "%.14f\n", (float)output[i]);
        }

        fprintf(fp_output, "\n"); // Xuống dòng sau mỗi kên
//    }
//    fclose(fp_output);

    printf("Processing completed. Outputs saved to dst_conv2.txt.\n");

    return 0;
}
