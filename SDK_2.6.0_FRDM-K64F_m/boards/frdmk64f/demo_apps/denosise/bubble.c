/*
 * Copyright (c) 2018-2020, Jianjia Ma
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2020-09-09     Jianjia Ma   The first version
 *
 *
 * This file is apart of NNoM examples, which aims to provide clear demo of every steps. 
 * Therefor, it is not optimized for neither space and speed. 
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "nnom.h"
#include "denoise_weights.h"

#include "mfcc.h"
#include "wav.h"

 // the bandpass filter coefficiences
#include "equalizer_coeff.h" 

//mcu file
#include "fsl_sd.h"
#include "fsl_debug_console.h"
#include "ff.h"
#include "diskio.h"
#include "fsl_sd_disk.h"
#include "board.h"

#include "fsl_sysmpu.h"
#include "pin_mux.h"
#include "clock_config.h"

#define NUM_FEATURES NUM_FILTER

#define _MAX(x, y) (((x) > (y)) ? (x) : (y))
#define _MIN(x, y) (((x) < (y)) ? (x) : (y))

#define NUM_CHANNELS 	1
#define SAMPLE_RATE 	16000
#define AUDIO_FRAME_LEN 512
   


// audio buffer for input
float audio_buffer[AUDIO_FRAME_LEN] = {0};
int16_t audio_buffer_16bit[AUDIO_FRAME_LEN] = {0};

// buffer for output
int16_t audio_buffer_filtered[AUDIO_FRAME_LEN/2] = { 0 };


// mfcc features and their derivatives
float mfcc_feature[NUM_FEATURES] = { 0 };
float mfcc_feature_prev[NUM_FEATURES] = { 0 };
float mfcc_feature_diff[NUM_FEATURES] = { 0 };
float mfcc_feature_diff_prev[NUM_FEATURES] = { 0 };
float mfcc_feature_diff1[NUM_FEATURES] = { 0 };
// features for NN
float nn_features[64] = {0};
int8_t nn_features_q7[64] = {0};

// NN results, which is the gains for each frequency band
float band_gains[NUM_FILTER] = {0};
float band_gains_prev[NUM_FILTER] = {0};

// 0db gains coefficient
float coeff_b[NUM_FILTER][NUM_COEFF_PAIR] = FILTER_COEFF_B;
float coeff_a[NUM_FILTER][NUM_COEFF_PAIR] = FILTER_COEFF_A;
// dynamic gains coefficient
float b_[NUM_FILTER][NUM_COEFF_PAIR] = {0};

/*!
 * @brief wait card insert function.
 */
static status_t sdcardWaitCardInsert(void);

/*******************************************************************************
 * Variables
 ******************************************************************************/
static FATFS g_fileSystem; /* File system object */
static FIL g_logFileObject;   /* File object */
static FIL g_inFileObject;
static FIL g_outFileObject;

/*! @brief SDMMC host detect card configuration */
static const sdmmchost_detect_card_t s_sdCardDetect = {
#ifndef BOARD_SD_DETECT_TYPE
    .cdType = kSDMMCHOST_DetectCardByGpioCD,
#else
    .cdType = BOARD_SD_DETECT_TYPE,
#endif
    .cdTimeOut_ms = (~0U),
};

/*! @brief SDMMC card power control configuration */
#if defined DEMO_SDCARD_POWER_CTRL_FUNCTION_EXIST
static const sdmmchost_pwr_card_t s_sdCardPwrCtrl = {
    .powerOn          = BOARD_PowerOnSDCARD,
    .powerOnDelay_ms  = 500U,
    .powerOff         = BOARD_PowerOffSDCARD,
    .powerOffDelay_ms = 0U,
};
#endif

// update the history
void y_h_update(float *y_h, uint32_t len)
{
	for (uint32_t i = len-1; i >0 ;i--)
		y_h[i] = y_h[i-1];
}

//  equalizer by multiple n order iir band pass filter. 
// y[i] = b[0] * x[i] + b[1] * x[i - 1] + b[2] * x[i - 2] - a[1] * y[i - 1] - a[2] * y[i - 2]...
void equalizer(float* x, float* y, uint32_t signal_len, float *b, float *a, uint32_t num_band, uint32_t num_order)
{
	// the y history for each band
	static float y_h[NUM_FILTER][NUM_COEFF_PAIR] = { 0 };
	static float x_h[NUM_COEFF_PAIR * 2] = { 0 };
	uint32_t num_coeff = num_order * 2 + 1;

	// i <= num_coeff (where historical x is involved in the first few points)
	// combine state and new data to get a continual x input. 
	memcpy(x_h + num_coeff, x, num_coeff * sizeof(float));
	for (uint32_t i = 0; i < num_coeff; i++)
	{
		y[i] = 0;
		for (uint32_t n = 0; n < num_band; n++)
		{
			y_h_update(y_h[n], num_coeff);
			y_h[n][0] = b[n * num_coeff] * x_h[i+ num_coeff];
			for (uint32_t c = 1; c < num_coeff; c++)
				y_h[n][0] += b[n * num_coeff + c] * x_h[num_coeff + i - c] - a[n * num_coeff + c] * y_h[n][c];
			y[i] += y_h[n][0];
		}
	}
	// store the x for the state of next round
	memcpy(x_h, &x[signal_len - num_coeff], num_coeff * sizeof(float));
	
	// i > num_coeff; the rest data not involed the x history
	for (uint32_t i = num_coeff; i < signal_len; i++)
	{
		y[i] = 0;
		for (uint32_t n = 0; n < num_band; n++)
		{
			y_h_update(y_h[n], num_coeff);
			y_h[n][0] = b[n * num_coeff] * x[i];
			for (uint32_t c = 1; c < num_coeff; c++)
				y_h[n][0] += b[n * num_coeff + c] * x[i - c] - a[n * num_coeff + c] * y_h[n][c];
			y[i] += y_h[n][0];
		}	
	}
}

// set dynamic gains. Multiple gains x b_coeff
void set_gains(float *b_in, float *b_out,  float* gains, uint32_t num_band, uint32_t num_order)
{
	uint32_t num_coeff = num_order * 2 + 1;
	for (uint32_t i = 0; i < num_band; i++)
		for (uint32_t c = 0; c < num_coeff; c++)
			b_out[num_coeff *i + c] = b_in[num_coeff * i + c] * gains[i]; // only need to set b. 
}

void quantize_data(float*din, int8_t *dout, uint32_t size, uint32_t int_bit)
{
	float limit = (1 << int_bit); 
	for(uint32_t i=0; i<size; i++)
		dout[i] = (int8_t)(_MAX(_MIN(din[i], limit), -limit) / limit * 127);
}

void log_values(float* value, uint32_t size)
{
	char line[16];
    UINT sizeToWrite;
	for (uint32_t i = 0; i < size; i++) {
		snprintf(line, 16, "%d.%d,", (int)value[i],((int)(value[i]*10)) % 10);
		f_write(&g_logFileObject,line, strlen(line), &sizeToWrite);
	}
	f_write(&g_logFileObject,"\n", 2, &sizeToWrite);
}


/*******************************************************************************
 * Code
 ******************************************************************************/

/*!
 * @brief Main function
 */
int main(void)
{
    FRESULT error;
    DIR directory; /* Directory object */
    FILINFO fileInformation;
    UINT bytesWritten;
    UINT bytesRead;
    const TCHAR driverNumberBuffer[3U] = {SDDISK + '0', ':', '/'};
    volatile bool failedFlag           = false;
    char ch                            = '0';
    BYTE work[FF_MAX_SS];

    BOARD_InitPins();
    BOARD_BootClockRUN();
    BOARD_InitDebugConsole();
    SYSMPU_Enable(SYSMPU, false);

    PRINTF("\r\nFATFS example to demonstrate how to use FATFS with SD card.\r\n");

    PRINTF("\r\nPlease insert a card into board.\r\n");

    if (sdcardWaitCardInsert() != kStatus_Success)
    {
        return -1;
    }

    if (f_mount(&g_fileSystem, driverNumberBuffer, 0U))
    {
        PRINTF("Mount volume failed.\r\n");
        return -1;
    }

#if (FF_FS_RPATH >= 2U)
    error = f_chdrive((char const *)&driverNumberBuffer[0U]);
    if (error)
    {
        PRINTF("Change drive failed.\r\n");
        return -1;
    }
#endif
    PRINTF("Any key to start\r\n");
    GETCHAR();
	wav_header_t wav_header; 
	size_t size;

	char* input_file = "sample.wav";
	char* output_file = "filtered_sample.wav";


	char* log_file = "log.csv";
	error = f_open(&g_logFileObject,log_file, FA_WRITE|FA_CREATE_ALWAYS);
	error = (error << 1) | (f_open(&g_inFileObject,input_file, FA_READ));
	error = (error << 1) | (f_open(&g_outFileObject,output_file, FA_WRITE|FA_CREATE_ALWAYS));
	if (error != FR_OK) 
	{
		printf("Cannot open wav files, default input:'%s'\n", input_file);
		printf("Or use command to specify input file: xxx.exe [input.wav] [output.wav]\n");
		return -1;
	}

		
	// read wav file header, copy it to the output file.  
    UINT sizeToread = 0;
    UINT sizeToWrite = 0;
	f_read(&g_inFileObject,&wav_header, sizeof(wav_header),&sizeToread);
	f_write(&g_outFileObject,&wav_header, sizeof(wav_header),&sizeToWrite);

	// lets jump to the "data" chunk of the WAV file.
	if (strncmp(wav_header.datachunk_id, "data", 4)){
		wav_chunk_t chunk = { .size= wav_header.datachunk_size};
		// find the 'data' chunk
		do {
			char* buf = malloc(chunk.size);
			f_read(&g_inFileObject,buf, chunk.size, &sizeToread);
			f_write(&g_outFileObject,buf, chunk.size, &sizeToWrite);
			free(buf);
			f_read(&g_inFileObject,&chunk, sizeof(wav_chunk_t), &sizeToread);
			f_write(&g_outFileObject,&chunk, sizeof(wav_chunk_t), &sizeToWrite);
		} while (strncmp(chunk.id, "data", 4));
	}
	
	// NNoM model
	nnom_model_t *model = nnom_model_create();
	
	// 26 features, 0 offset, 26 bands, 512fft, 0 preempha, attached_energy_to_band0
	mfcc_t * mfcc = mfcc_create(NUM_FEATURES, 0, NUM_FEATURES, 512, 0, true);

	printf("\nProcessing file: %s\r\n", input_file);
    uint32_t wav_sum = 0;
    uint32_t wav_result = 0;
     uint32_t pre_wav_result = 0;
	while(1) 
    {
		// move buffer (50%) overlapping, move later 50% to the first 50, then fill 
		memcpy(audio_buffer_16bit, &audio_buffer_16bit[AUDIO_FRAME_LEN/2], AUDIO_FRAME_LEN/2*sizeof(int16_t));

		// now read the new data
		f_read(&g_inFileObject,&audio_buffer_16bit[AUDIO_FRAME_LEN / 2], AUDIO_FRAME_LEN / 2 * sizeof(int16_t), &sizeToread);
        if(sizeToread == 0)
            break;
        wav_sum += sizeToread;
        wav_result = (int)((wav_sum/(float)wav_header.datachunk_size)*100);
        if(wav_result != pre_wav_result)
        {
            pre_wav_result = wav_result;
            PRINTF(">");
            PRINTF("%d%%", wav_result);
            PRINTF("\b\b\b");
        }
		
		// get mfcc
		mfcc_compute(mfcc, audio_buffer_16bit, mfcc_feature);
		
//log_values(mfcc_feature, NUM_FEATURES, flog);

		// get the first and second derivative of mfcc
		for(uint32_t i=0; i< NUM_FEATURES; i++)
		{
			mfcc_feature_diff[i] = mfcc_feature[i] - mfcc_feature_prev[i];
			mfcc_feature_diff1[i] = mfcc_feature_diff[i] - mfcc_feature_diff_prev[i];
		}
		memcpy(mfcc_feature_prev, mfcc_feature, NUM_FEATURES * sizeof(float));
		memcpy(mfcc_feature_diff_prev, mfcc_feature_diff, NUM_FEATURES * sizeof(float));
		
		// combine MFCC with derivatives 
		memcpy(nn_features, mfcc_feature, NUM_FEATURES*sizeof(float));
		memcpy(&nn_features[NUM_FEATURES], mfcc_feature_diff, 10*sizeof(float));
		memcpy(&nn_features[NUM_FEATURES+10], mfcc_feature_diff1, 10*sizeof(float));

//log_values(nn_features, NUM_FEATURES+20, flog);
		
		// quantise them using the same scale as training data (in keras), by 2^n. 
		quantize_data(nn_features, nn_features_q7, NUM_FEATURES+20, 3);
		
		// run the mode with the new input
		memcpy(nnom_input_data, nn_features_q7, sizeof(nnom_input_data));
		model_run(model);
		
		// read the result, convert it back to float (q0.7 to float)
		for(int i=0; i< NUM_FEATURES; i++)
			band_gains[i] = (float)(nnom_output_data[i]) / 127.f;

        log_values(band_gains, NUM_FILTER);
		
		// one more step, limit the change of gians, to smooth the speech, per RNNoise paper
		for(int i=0; i< NUM_FEATURES; i++)
			band_gains[i] = _MAX(band_gains_prev[i]*0.8f, band_gains[i]); 
		memcpy(band_gains_prev, band_gains, NUM_FEATURES *sizeof(float));
		
		// apply the dynamic gains to each frequency band. 
		set_gains((float*)coeff_b, (float*)b_, band_gains, NUM_FILTER, NUM_ORDER);

		// convert 16bit to float for equalizer
		for (int i = 0; i < AUDIO_FRAME_LEN/2; i++)
			audio_buffer[i] = audio_buffer_16bit[i + AUDIO_FRAME_LEN / 2] / 32768.f;
				
		// finally, we apply the equalizer to this audio frame to denoise
		equalizer(audio_buffer, &audio_buffer[AUDIO_FRAME_LEN / 2], AUDIO_FRAME_LEN/2, (float*)b_,(float*)coeff_a, NUM_FILTER, NUM_ORDER);

		// convert the filtered signal back to int16
		for (int i = 0; i < AUDIO_FRAME_LEN / 2; i++)
			audio_buffer_filtered[i] = audio_buffer[i + AUDIO_FRAME_LEN / 2] * 32768.f *0.6f; 
		
		// write the filtered frame to WAV file. 
		f_write(&g_outFileObject,audio_buffer_filtered, 256*sizeof(int16_t),&sizeToWrite);
	}

	// print some model info
	model_io_format(model);
	model_stat(model);
	model_delete(model);

	f_close(&g_inFileObject);
	f_close(&g_outFileObject);
	f_close(&g_logFileObject);

	printf("\r\nNoisy signal '%s' has been de-noised by NNoM.\r\nThe output is saved to '%s'.\r\n", input_file, output_file);
	return 0;

    

    while (true)
    {
    }
}

static status_t sdcardWaitCardInsert(void)
{
    /* Save host information. */
    g_sd.host.base           = SD_HOST_BASEADDR;
    g_sd.host.sourceClock_Hz = SD_HOST_CLK_FREQ;
    /* card detect type */
    g_sd.usrParam.cd = &s_sdCardDetect;
#if defined DEMO_SDCARD_POWER_CTRL_FUNCTION_EXIST
    g_sd.usrParam.pwr = &s_sdCardPwrCtrl;
#endif
    /* SD host init function */
    if (SD_HostInit(&g_sd) != kStatus_Success)
    {
        PRINTF("\r\nSD host init fail\r\n");
        return kStatus_Fail;
    }
    /* power off card */
    SD_PowerOffCard(g_sd.host.base, g_sd.usrParam.pwr);
    /* wait card insert */
    if (SD_WaitCardDetectStatus(SD_HOST_BASEADDR, &s_sdCardDetect, true) == kStatus_Success)
    {
        PRINTF("\r\nCard inserted.\r\n");
        /* power on the card */
        SD_PowerOnCard(g_sd.host.base, g_sd.usrParam.pwr);
    }
    else
    {
        PRINTF("\r\nCard detect fail.\r\n");
        return kStatus_Fail;
    }

    return kStatus_Success;
}



#if 0
int main(int argc, char* argv[])
{
	wav_header_t wav_header; 
	size_t size;

	char* input_file = "sample.wav";
	char* output_file = "filtered_sample.wav";
	FILE* src_file;
	FILE* des_file;

	char* log_file = "log.csv";
	FILE* flog = fopen(log_file, "wb");

	// if user has specify input and output files. 
	if (argc > 1)
		input_file = argv[1];
	if (argc > 2)
		output_file = argv[2];

	src_file = fopen(input_file, "rb");
	des_file = fopen(output_file, "wb");
	if (src_file == NULL) 
	{
		printf("Cannot open wav files, default input:'%s'\n", input_file);
		printf("Or use command to specify input file: xxx.exe [input.wav] [output.wav]\n");
		return -1;
	}
	if (des_file == NULL)
	{
		fclose(src_file); 
		return -1; 
	}
		
	// read wav file header, copy it to the output file.  
	f_read(&wav_header, sizeof(wav_header), 1, src_file);
	f_write(&wav_header, sizeof(wav_header), 1, des_file);

	// lets jump to the "data" chunk of the WAV file.
	if (strncmp(wav_header.datachunk_id, "data", 4)){
		wav_chunk_t chunk = { .size= wav_header.datachunk_size};
		// find the 'data' chunk
		do {
			char* buf = malloc(chunk.size);
			f_read(buf, chunk.size, 1, src_file);
			f_write(buf, chunk.size, 1, des_file);
			free(buf);
			f_read(&chunk, sizeof(wav_chunk_t), 1, src_file);
			f_write(&chunk, sizeof(wav_chunk_t), 1, des_file);
		} while (strncmp(chunk.id, "data", 4));
	}
	
	// NNoM model
	nnom_model_t *model = model = nnom_model_create();
	
	// 26 features, 0 offset, 26 bands, 512fft, 0 preempha, attached_energy_to_band0
	mfcc_t * mfcc = mfcc_create(NUM_FEATURES, 0, NUM_FEATURES, 512, 0, true);

	printf("\nProcessing file: %s\n", input_file);
	while(1) {
		// move buffer (50%) overlapping, move later 50% to the first 50, then fill 
		memcpy(audio_buffer_16bit, &audio_buffer_16bit[AUDIO_FRAME_LEN/2], AUDIO_FRAME_LEN/2*sizeof(int16_t));

		// now read the new data
		size = f_read(&audio_buffer_16bit[AUDIO_FRAME_LEN / 2], AUDIO_FRAME_LEN / 2 * sizeof(int16_t), 1, src_file);
		if(size == 0)
			break;
		
		// get mfcc
		mfcc_compute(mfcc, audio_buffer_16bit, mfcc_feature);
		
//log_values(mfcc_feature, NUM_FEATURES, flog);

		// get the first and second derivative of mfcc
		for(uint32_t i=0; i< NUM_FEATURES; i++)
		{
			mfcc_feature_diff[i] = mfcc_feature[i] - mfcc_feature_prev[i];
			mfcc_feature_diff1[i] = mfcc_feature_diff[i] - mfcc_feature_diff_prev[i];
		}
		memcpy(mfcc_feature_prev, mfcc_feature, NUM_FEATURES * sizeof(float));
		memcpy(mfcc_feature_diff_prev, mfcc_feature_diff, NUM_FEATURES * sizeof(float));
		
		// combine MFCC with derivatives 
		memcpy(nn_features, mfcc_feature, NUM_FEATURES*sizeof(float));
		memcpy(&nn_features[NUM_FEATURES], mfcc_feature_diff, 10*sizeof(float));
		memcpy(&nn_features[NUM_FEATURES+10], mfcc_feature_diff1, 10*sizeof(float));

//log_values(nn_features, NUM_FEATURES+20, flog);
		
		// quantise them using the same scale as training data (in keras), by 2^n. 
		quantize_data(nn_features, nn_features_q7, NUM_FEATURES+20, 3);
		
		// run the mode with the new input
		memcpy(nnom_input_data, nn_features_q7, sizeof(nnom_input_data));
		model_run(model);
		
		// read the result, convert it back to float (q0.7 to float)
		for(int i=0; i< NUM_FEATURES; i++)
			band_gains[i] = (float)(nnom_output_data[i]) / 127.f;

        log_values(band_gains, NUM_FILTER, flog);
		
		// one more step, limit the change of gians, to smooth the speech, per RNNoise paper
		for(int i=0; i< NUM_FEATURES; i++)
			band_gains[i] = _MAX(band_gains_prev[i]*0.8f, band_gains[i]); 
		memcpy(band_gains_prev, band_gains, NUM_FEATURES *sizeof(float));
		
		// apply the dynamic gains to each frequency band. 
		set_gains((float*)coeff_b, (float*)b_, band_gains, NUM_FILTER, NUM_ORDER);

		// convert 16bit to float for equalizer
		for (int i = 0; i < AUDIO_FRAME_LEN/2; i++)
			audio_buffer[i] = audio_buffer_16bit[i + AUDIO_FRAME_LEN / 2] / 32768.f;
				
		// finally, we apply the equalizer to this audio frame to denoise
		equalizer(audio_buffer, &audio_buffer[AUDIO_FRAME_LEN / 2], AUDIO_FRAME_LEN/2, (float*)b_,(float*)coeff_a, NUM_FILTER, NUM_ORDER);

		// convert the filtered signal back to int16
		for (int i = 0; i < AUDIO_FRAME_LEN / 2; i++)
			audio_buffer_filtered[i] = audio_buffer[i + AUDIO_FRAME_LEN / 2] * 32768.f *0.6f; 
		
		// write the filtered frame to WAV file. 
		f_write(audio_buffer_filtered, 256*sizeof(int16_t), 1, des_file);
	}

	// print some model info
	model_io_format(model);
	model_stat(model);
	model_delete(model);

	fclose(flog);
	fclose(src_file);
	fclose(des_file);

	printf("\nNoisy signal '%s' has been de-noised by NNoM.\nThe output is saved to '%s'.\n", input_file, output_file);
	return 0;
}
#endif