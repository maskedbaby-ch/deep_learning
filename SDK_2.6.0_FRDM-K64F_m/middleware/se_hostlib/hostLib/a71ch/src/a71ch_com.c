/**
* @file a71ch_com.c
* @author NXP Semiconductors
* @version 1.0
* @par License
* Copyright 2016 NXP
*
* This software is owned or controlled by NXP and may only be used
* strictly in accordance with the applicable license terms.  By expressly
* accepting such terms or by downloading, installing, activating and/or
* otherwise using the software, you are agreeing that you have read, and
* that you agree to comply with and are bound by, such license terms.  If
* you do not agree to be bound by the applicable license terms, then you
* may not retain, install, activate or otherwise use the software.
*
* @par Description
* Implementation of basic communication functionality between Host and A71CH.
* @par History
* 1.0   1-oct-2016 : Initial version
*
*****************************************************************************/

#if defined(SSS_USE_FTR_FILE)
#include "fsl_sss_ftr.h"
#else
#include "fsl_sss_ftr_default.h"
#endif

#include <a71ch_const.h>
#include <smCom.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "sm_api.h"
#include "sm_apdu.h"
#include "sm_errors.h"
#include "sm_printf.h"
#include "sm_types.h"

#include "nxLog_smCom.h"

#if SSS_HAVE_SSCP
#include "fsl_sscp_a71ch.h"
#endif

//Also do select after opening the connection
#define OPEN_AND_SELECT 0


#ifdef TDA8029_UART
#include "smComAlpar.h"
#include "smUart.h"
#endif
#if defined(I2C)
#include "smComSCI2C.h"
#endif
#if defined(SPI)
#include "smComSCSPI.h"
#endif
#if defined(PCSC)
#include "smComPCSC.h"
#endif
#if defined(IPC)
#include "smComIpc.h"
#endif
#if defined(SMCOM_JRCP_V1)
#include "smComSocket.h"
#endif
#if defined(SMCOM_JRCP_V2)
#include "smComJRCP.h"
#endif
#if defined(RJCT_VCOM)
#include "smComSerial.h"
#endif
#if defined(T1oI2C)
#include "smComT1oI2C.h"
#endif
#if defined(SMCOM_PN7150)
#include "smComPN7150.h"
#endif
#if defined(SMCOM_THREAD)
#include "smComThread.h"
#endif

#include "global_platf.h"

/// @cond Optional diagnostics functionality
// #define FLOW_VERBOSE
#ifdef FLOW_VERBOSE
#define FPRINTF(...) printf(__VA_ARGS__)
#else
#define FPRINTF(...)
#endif
/// @endcond

#if defined(SMCOM_JRCP_V1) || defined(SMCOM_JRCP_V2)
static U16 getSocketParams(
    const char *arg, U8 *szServer, U16 szServerLen, unsigned int *port)
{
    // the IP address is in format a.b.c.d:port, e.g. 10.0.0.1:8080
    int nSuccess;

    if (strlen(arg) >= szServerLen) {
        return ERR_BUF_TOO_SMALL;
    }

    // First attempt at parsing: server IP-address passed, sscanf will return 2 upon successfull parsing
    nSuccess =
        sscanf(arg, "%15[0-9.]:%5u[0-9]", szServer, (unsigned int *)port);
    if (nSuccess == 2) {
        return SW_OK;
    }
    else {
        // Second attempt at parsing: server name passed instead of IP-address
        unsigned int i;
        int fColonFound = 0;

        for (i = 0; i < strlen(arg); i++) {
            if (arg[i] == ':') {
                szServer[i] = 0;
                fColonFound = 1;
                // PRINTF("servername: %s\r\n", szServer);
                break;
            }
            else {
                szServer[i] = arg[i];
            }
        }

        if ((fColonFound == 1) && (i != 0)) {
            nSuccess = sscanf(&arg[i], ":%5u[0-9]", (unsigned int *)port);
            if (nSuccess == 1) {
                return SW_OK;
            }
        }
    }
    return ERR_NO_VALID_IP_PORT_PATTERN;
}

/**
* Establishes communication with the Security Module via a Remote JC Terminal Server
* (RJCT-Server).
* Next it will invoke ::SM_Connect and select the A71CH applet on the Secure Module
*
* \note Because connecting via an RJCT-server requires an extra parameter (the server IP:Port)
* an additional function is required on top of ::SM_Connect
*
* @param[in,out] connectString ip:port as string
* @param[in,out] commState
* @param[in,out] atr
* @param[in,out] atrLen
*
* @retval ::SW_OK Upon successful execution
*/
U16 SM_RjctConnectSocket(
    const char *connectString, SmCommState_t *commState, U8 *atr, U16 *atrLen)
{
    U8 szServer[128];
    U16 szServerLen = sizeof(szServer);
    U16 rv = 0;
    unsigned int port = 0;
	char hostname[32];

#ifndef A71_IGNORE_PARAM_CHECK
    if ((connectString == NULL) || (commState == NULL) || (atr == NULL) || (atrLen == 0)) {
        return ERR_API_ERROR;
    }
#endif

    rv = getSocketParams(
        connectString, szServer, szServerLen, (unsigned int *)&port);

#if defined(SMCOM_JRCP_V1)
    FPRINTF("Connection to secure element over socket to %s\r\n", connectString);
    if (rv != SW_OK) {
        return rv;
    }
	// NOTE-MMA: The usage of the sss type kType_SE_Conn_Type_JRCP_V1 leads to a circular
	// dependency regarding the inclusion of header files.
    // if (commState->connType == kType_SE_Conn_Type_JRCP_V1) {
        rv = smComSocket_Open(szServer, (U16)port, atr, atrLen);
    // }

#endif
#if defined(SMCOM_JRCP_V2)
    if (commState->connType == kType_SE_Conn_Type_JRCP_V2) {
		strncpy(hostname, connectString, strlen(connectString));
        rv = smComJRCP_Open(strtok(hostname, ":"), port);
    }

#endif
    if (rv != SMCOM_OK) {
        LOG_E("Error on smComSocket_Open: 0x%04X\r\n", rv);
        return rv;
    }

    rv = SM_Connect(commState, atr, atrLen);
    return rv;
}
#endif /* defined(SMCOM_JRCP_V1) || defined (SMCOM_JRCP_V2) */

#ifdef RJCT_VCOM
U16 SM_RjctConnectVCOM(
    const char *connectString, SmCommState_t *commState, U8 *atr, U16 *atrLen)
{
    U32 status;

#ifndef A71_IGNORE_PARAM_CHECK
    if ((connectString == NULL) || (commState == NULL) || (atr == NULL) || (atrLen == 0)) {
        return ERR_API_ERROR;
    }
#endif

    status = smComVCom_Open(connectString);

    if (status == 0) {
        status = smComVCom_GetATR(atr, atrLen);
        if (status == 0) {
            status = (U16)SM_Connect(commState, atr, atrLen);
        }
    }
    else {
        *atrLen = 0;
    }

    return (U16)status;
}
#endif // RJCT_VCOM

U16 SM_RjctConnect(
    const char *connectString, SmCommState_t *commState, U8 *atr, U16 *atrLen)
{
#if RJCT_VCOM || SMCOM_JRCP_V1 || SMCOM_JRCP_V2
    bool is_socket = FALSE;
    bool is_vcom = FALSE;
    AX_UNUSED_ARG(is_socket);
    AX_UNUSED_ARG(is_vcom);
#endif

#if RJCT_VCOM
    if (0 == strncmp("COM", connectString, sizeof("COM") - 1)) {
        is_vcom = TRUE;
    }
    else if (0 ==
             strncmp("\\\\.\\COM", connectString, sizeof("\\\\.\\COM") - 1)) {
        is_vcom = TRUE;
    }
    else if (0 == strncmp("/tty/", connectString, sizeof("/tty/") - 1)) {
        is_vcom = TRUE;
    }
    else if (0 == strncmp("/dev/tty.", connectString, sizeof("/dev/tty.") - 1)) {
        is_vcom = TRUE;
    }
#endif
#if SMCOM_JRCP_V1 || SMCOM_JRCP_V2
    if (NULL != strchr(connectString, ':')) {
        is_socket = TRUE;
    }
#endif
#if RJCT_VCOM
    if (is_vcom) {
        return SM_RjctConnectVCOM(connectString, commState, atr, atrLen);
    }
#endif
#if SMCOM_JRCP_V1 || SMCOM_JRCP_V2
    if (is_socket) {
        return SM_RjctConnectSocket(connectString, commState, atr, atrLen);
    }
#endif
    sm_printf(CONSOLE,
        "Can not use connectString='%s' in the current build configuration.\n\tPlease select correct smCom interface and re-compile!\n",
        connectString);
    return ERR_NO_VALID_IP_PORT_PATTERN;
}

/**
* Establishes the communication with the Security Module (SM) at the link level and
* selects the A71CH applet on the SM. The physical communication layer used (e.g. I2C)
* is determined at compilation time.
*
* @param[in,out] commState
* @param[in,out] atr
* @param[in,out] atrLen
*
* @retval ::SW_OK Upon successful execution
*/
U16 SM_Connect(SmCommState_t *commState, U8 *atr, U16 *atrLen)
{
    U16 sw = SW_OK;
#if !defined(IPC)
    unsigned char appletName[] = APPLET_NAME;
    U16 selectResponseDataLen = 0;
    U8 selectResponseData[256] = {0};
    U16 uartBR = 0;
    U16 t1BR = 0;
#endif
#ifdef TDA8029_UART
    U32 status = 0;
#endif

#ifndef A71_IGNORE_PARAM_CHECK
    if ((commState == NULL) || (atr == NULL) || (atrLen == 0)) {
        return ERR_API_ERROR;
    }
#endif

#ifdef TDA8029_UART
    if ((*atrLen) <= 33) return ERR_API_ERROR;

    smComAlpar_Init();
    status = smComAlpar_AtrT1Configure(
        ALPAR_T1_BAUDRATE_MAX, atr, atrLen, &uartBR, &t1BR);
    if (status != SMCOM_ALPAR_OK) {
        commState->param1 = 0;
        commState->param2 = 0;
        FPRINTF("smComAlpar_AtrT1Configure failed: 0x%08X\r\n", status);
        return ERR_CONNECT_LINK_FAILED;
    }
#elif defined SMCOM_PN7150
    sw = smComPN7150_Open(0, 0x00, atr, atrLen);
#elif defined(I2C)
    sw = smComSCI2C_Open(ESTABLISH_SCI2C, 0x00, atr, atrLen);
#elif defined(SPI)
    smComSCSPI_Init(ESTABLISH_SCI2C, 0x00, atr, atrLen);
#elif defined(PCSC)
    sw = smComPCSC_Open(0, atr, atrLen);
#elif defined(IPC)
    sw = smComIpc_Open(atr,
        atrLen,
        &(commState->hostLibVersion),
        &(commState->appletVersion),
        &(commState->sbVersion));
#elif defined(T1oI2C)
    sw = smComT1oI2C_Open(ESE_MODE_NORMAL, 0x00, atr, atrLen);
#elif defined(SMCOM_JRCP_V1) || defined(SMCOM_JRCP_V2)
    if (atrLen != NULL)
        *atrLen = 0;
    AX_UNUSED_ARG(atr);
    AX_UNUSED_ARG(atrLen);
#elif defined(RJCT_VCOM)
#elif defined(SMCOM_THREAD)
    sw = smComThread_Open(atr, atrLen);
#endif // TDA8029_UART


#if !defined(IPC)
    commState->param1 = t1BR;
    commState->param2 = uartBR;
    commState->hostLibVersion = (AX_HOST_LIB_MAJOR << 8) + AX_HOST_LIB_MINOR;
    commState->appletVersion = 0xFFFF;
    commState->sbVersion = 0xFFFF;

    if (sw == SMCOM_OK) {
        selectResponseDataLen = sizeof(selectResponseData);
        /* CARD */
        sw = GP_Select(
            (U8 *)&appletName, 0, selectResponseData, &selectResponseDataLen);
        sw = GP_Select((U8 *)&appletName,
            APPLET_NAME_LEN,
            selectResponseData,
            &selectResponseDataLen);

        if (sw == SW_FILE_NOT_FOUND) {
            // Applet can not be selected (most likely it is simply not installed)
            LOG_E("Can not select Applet=%s'", SE_NAME);
            LOG_MAU8_E("Failed (SW_FILE_NOT_FOUND) selecting Applet. ",
                appletName, APPLET_NAME_LEN);
            return sw;
        }
        else if (sw != SW_OK) {
            sw = ERR_CONNECT_SELECT_FAILED;
        }
        else {
#ifdef FLOW_VERBOSE
            printf("selectResponseDataLen: %d\r\n", selectResponseDataLen);
            {
                int i = 0;
                for (i = 0; i < selectResponseDataLen; i++) {
                    printf("0x%02X:", selectResponseData[i]);
                }
                printf("\r\n");
            }
#endif
#if SSS_HAVE_A71CH || SSS_HAVE_SE050_EAR_CH
            if (selectResponseDataLen >= 2) {
                commState->appletVersion =
                    (selectResponseData[0] << 8) + selectResponseData[1];
                if (selectResponseDataLen == 4) {
                    commState->sbVersion =
                        (selectResponseData[2] << 8) + selectResponseData[3];
                }
                else if (selectResponseDataLen == 2) {
                    commState->sbVersion = 0x0000;
                }
            }
            else {
#if SSS_HAVE_SE050_EAR
                commState->appletVersion = 0;
                commState->sbVersion = 0;
#else
                sw = ERR_CONNECT_SELECT_FAILED;
#endif
            }
#elif SSS_HAVE_A71CL
            if (selectResponseDataLen == 0) {
                commState->appletVersion = 0;
                commState->sbVersion = 0x0000;
            }
#endif
#if SSS_HAVE_SE05X
            if (selectResponseDataLen == 5 || selectResponseDataLen == 4) {
                // 2.2.4 returns 4 bytes, 2.2.4.[A,B,C]
                // 2.3.0 returns 5 bytes, 2.3.0.[v1].[v2]
                commState->appletVersion = 0;
                commState->appletVersion |= selectResponseData[0];
                commState->appletVersion <<= 8;
                commState->appletVersion |= selectResponseData[1];
                commState->appletVersion <<= 8;
                commState->appletVersion |= selectResponseData[2];
                commState->appletVersion <<= 8;
                // commState->appletVersion |= selectResponseData[3];
                commState->sbVersion = 0x0000;
            }
            else
            {

            }
#endif
        }
    }
#endif // !defined(IPC)
    return sw;
}

/**
 * Closes the communication with the Security Module
 * A new connection can be established by calling ::SM_Connect
 *
 * @param[in] mode Specific information that may be required on the link layer
 *
 * @retval ::SW_OK Upon successful execution
 */
U16 SM_Close(U8 mode)
{
    U16 sw = SW_OK;

#if defined(I2C)
    sw = smComSCI2C_Close(mode);
#endif
#if defined(SPI)
    sw = smComSCSPI_Close(mode);
#endif
#if defined(PCSC)
    sw = smComPCSC_Close(mode);
#endif
#if defined(IPC)
    AX_UNUSED_ARG(mode);
    sw = smComIpc_Close();
#endif
#if defined(T1oI2C)
    sw = smComT1oI2C_Close(mode);
#endif
#if defined(SMCOM_JRCP_V1)
    AX_UNUSED_ARG(mode);
    sw = smComSocket_Close();
#endif
#if defined(SMCOM_JRCP_V2)
    AX_UNUSED_ARG(mode);
    sw = smComJRCP_Close(mode);
#endif
#if defined(RJCT_VCOM)
    AX_UNUSED_ARG(mode);
    sw = smComVCom_Close();
#endif
#if defined(SMCOM_THREAD)
    AX_UNUSED_ARG(mode);
    sw = smComThread_Close();
#endif

    return sw;
}

/**
 * Sends the command APDU to the Secure Module and retrieves the response APDU.
 * The latter consists of the concatenation of the response data (possibly none) and the status word (2 bytes).
 *
 * The command APDU and response APDU are not interpreted by the host library.
 *
 * The command/response APDU sizes must lay within the APDU size limitations
 *
 * @param[in] cmd   command APDU
 * @param[in] cmdLen length (in byte) of \p cmd
 * @param[in,out] resp  response APDU (response data || response status word)
 * @param[in,out] respLen IN: Length of resp buffer (\p resp) provided; OUT: effective length of response retrieved.
 *
 * @retval ::SW_OK Upon successful execution
 */
U16 SM_SendAPDU(U8 *cmd, U16 cmdLen, U8 *resp, U16 *respLen)
{
    U32 status = 0;
    U32 respLenLocal;

#ifndef A71_IGNORE_PARAM_CHECK
    if ((cmd == NULL) || (resp == NULL) || (respLen == NULL)) {
        return ERR_API_ERROR;
    }
#endif

    respLenLocal = *respLen;

    status = smCom_TransceiveRaw(cmd, cmdLen, resp, &respLenLocal);
    *respLen = (U16)respLenLocal;

    return (U16)status;
}

#if defined(IPC)
U16 SM_LockChannel()
{
    return smComIpc_LockChannel();
}

U16 SM_UnlockChannel()
{
    return smComIpc_UnlockChannel();
}
#endif
