/*
 * Copyright (C) 2011-2018 Intel Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of Intel Corporation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */


#include <stdarg.h>
#include <stdio.h>      /* vsnprintf */
#include <math.h> //can I use math.h in the enclave?
#include "Enclave.h"

#ifndef NOENCLAVE
# include "Enclave_t.h"  /* print_string */
#endif

/* 
 * printf: 
 *   Invokes OCALL to display the enclave buffer to the terminal.
 */
#ifndef NOENCLAVE 
void printf(const char *fmt, ...)
{
    char buf[BUFSIZ] = {'\0'};
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
}
#endif

void printf_helloworld()
{
    printf("Hello World\n");
}

void add_in_enclave(int num1, int num2, int *sum, int len)
{
    printf("Entered add_in_enclave ecall\n");
	printf(sum);
    *sum = num1 + num2;  
    printf("Computed %i + %i = %i\n", num1, num2, *sum);

    return;
}

float forward_in_enclave(float * x, int x_width, int x_height)
{
	x = pool_in_enclave( x, 2, x_width, x_height);
	x = relu_in_enclave( x, x_width, x_height);
	return x;
}


float relu_in_enclave(float * x, int x_width, int x_height)
{
	printf("Entered relu_in_enclave\n");
	//x_height = ; //TODO: figure out x_height
	//x_width = ;//TODO: figure out x_width
	for( int i = 0; i< x_height; ++i ){
    	for (int j=0; j< x_width; ++j) {
        	if (x[i][j] <= 0){
            	in[i][j]= 0.0;
        	}
        	else in[i][j]= x[i][j];
		}
    }
    return  in; //pointer arithmatic?????
}
 	
}
float pool_in_enclave(float * in, int dim, int out_width, int out_height)
{
	printf("Entered pool_in_enclave\n")
	//how to get size from a pointer?
	//int out_height = ; //TODO: figure out out_height
	//int out_width = ; //TODO: figure out out_width
	int pool_y = dim;
	int pool_x = dim;
	float out[out_width][out_height];
	for (size_t y = 0; y < out_height; ++y) {
    	for (size_t x = 0; x < out_width; ++x) {
        	for (size_t i = 0; i < pool_y; ++i) {
            	for (size_t j = 0; j < pool_x; ++j) {
              
                    float value = in[y * pool_y + i][x * pool_x + j];
                    out[y][x] = max(out[y][x], value);
                
            	}
        	}
    	}
	}
	return out;
}
float softmax_in_enclave(float * input, int x_size, int y_size)
{
	printf("Entered softmax_in_enclave\n")
	int i;
	int j;
	double m, sum, constant;

	m = -INFINITY;
	for (i = 0; i < x_size; ++i) {
	for (j = 0, j < y_size; ++j){
		if (m < input[i][j]) {
			m = input[i][j];
		}
	}
	}

	sum = 0.0;
	for (i = 0; i < x_size; ++i) {
	for (j = 0; j < y_size; ++j) {

		sum += exp(input[i][j] - m);
	}
	}

	constant = m + log(sum);
	for (i = 0; i < x_size; ++i) {
	for (j = 0; j < y_size; ++j) {
		input[i][j] = exp(input[i][j] - constant);
	}
	}
	return input;
}


