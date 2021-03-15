/*
 * Copyright (c) 2015, Freescale Semiconductor, Inc.
 * Copyright 2016-2017 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "fsl_debug_console.h"
#include "board.h"
#include "fsl_common.h"
#include "pin_mux.h"
#include "fsl_gpio.h"
#include "fsl_port.h"
#include "nnom_port.h"
#include "nnom.h"
#include "weights.h"
#include "image.h"
/*******************************************************************************
 * Definitions
 ******************************************************************************/


/*******************************************************************************
 * Prototypes
 ******************************************************************************/

/*******************************************************************************
 * Variables
 ******************************************************************************/
nnom_model_t *model;
uint8_t temp[28*28]={0};
const char codeLib[] = "@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'.   ";
/*******************************************************************************
 * Code
 ******************************************************************************/
void print_img(int8_t * buf)
{
    for(int y = 0; y < 28; y++) 
	{
        for (int x = 0; x < 28; x++) 
		{
            int index =  69 / 127.0 * (127 - buf[y*28+x]); 
			if(index > 69) index =69;
			if(index < 0) index = 0;
            PRINTF("%c",codeLib[index]);
			PRINTF("%c",codeLib[index]);
        }
        PRINTF("\r\n");
    }
}


void mnist_pic(uint8_t *temp)
{
	float prob;
    uint32_t predic_label;
	PRINTF("\nprediction start.. \r\n");

	// copy data and do prediction
	memcpy(nnom_input_data, (int8_t*)temp, 784);
	nnom_predict(model, &predic_label, &prob);

	//print original image to console
	print_img((int8_t *)temp);
	PRINTF("\r\nPredicted label: %d\n", predic_label);
	PRINTF("\r\nProbability: %d%%\n", (int)(prob*100));
}

int main(void)
{
//    uint8_t ch;
//    uint16_t i = 0;
    /* Board pin, clock, debug console init */
    BOARD_InitPins();
    BOARD_BootClockRUN();
    BOARD_InitDebugConsole();
    /* Print a note to terminal */
    model = nnom_model_create();
	// dummy run
	model_run(model);
    while(1)
    {
        PRINTF("\r\n Send picture by serial\r\n");    
        DbgConsole_ReadLine(temp,784);
        PRINTF("\r\n Got picture\r\n");   
        mnist_pic(temp);
    }
}
