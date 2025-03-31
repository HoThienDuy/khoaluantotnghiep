// Copyright (C) 2024 Advanced Micro Devices, Inc
//
// SPDX-License-Identifier: MIT

//typedef unsigned char U8;

#pragma once


#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "ap_fixed.h"
#define MAX_IMAGE_WIDTH     128
#define MAX_IMAGE_HEIGHT    128
#define MAX_IMAGE_WIDTH_2     63
#define MAX_IMAGE_HEIGHT_2   63
#define FILTER_V_SIZE		3
#define FILTER_H_SIZE		3

//#ifndef MIN
//#define MIN(a,b) ((a<b)?a:b)
//#endif
//#ifndef MAX
//#define MAX(a,b) ((a<b)?b:a)
//#endif



typedef ap_fixed<16,2,AP_RND,AP_SAT> DTYPE_OUT;
typedef ap_fixed<16,2,AP_RND,AP_SAT> DTYPE_IN;

struct window {
	DTYPE_OUT pix [FILTER_V_SIZE][FILTER_H_SIZE];
};
//struct window1 {
//	DTYPE_OUT pix [8][FILTER_V_SIZE][FILTER_H_SIZE];
//};

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
);

}


