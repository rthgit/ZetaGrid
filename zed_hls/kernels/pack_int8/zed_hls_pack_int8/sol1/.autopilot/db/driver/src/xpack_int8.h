// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef XPACK_INT8_H
#define XPACK_INT8_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xpack_int8_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
#ifdef SDT
    char *Name;
#else
    u16 DeviceId;
#endif
    u64 Control_BaseAddress;
} XPack_int8_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XPack_int8;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XPack_int8_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XPack_int8_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XPack_int8_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XPack_int8_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
#ifdef SDT
int XPack_int8_Initialize(XPack_int8 *InstancePtr, UINTPTR BaseAddress);
XPack_int8_Config* XPack_int8_LookupConfig(UINTPTR BaseAddress);
#else
int XPack_int8_Initialize(XPack_int8 *InstancePtr, u16 DeviceId);
XPack_int8_Config* XPack_int8_LookupConfig(u16 DeviceId);
#endif
int XPack_int8_CfgInitialize(XPack_int8 *InstancePtr, XPack_int8_Config *ConfigPtr);
#else
int XPack_int8_Initialize(XPack_int8 *InstancePtr, const char* InstanceName);
int XPack_int8_Release(XPack_int8 *InstancePtr);
#endif

void XPack_int8_Start(XPack_int8 *InstancePtr);
u32 XPack_int8_IsDone(XPack_int8 *InstancePtr);
u32 XPack_int8_IsIdle(XPack_int8 *InstancePtr);
u32 XPack_int8_IsReady(XPack_int8 *InstancePtr);
void XPack_int8_EnableAutoRestart(XPack_int8 *InstancePtr);
void XPack_int8_DisableAutoRestart(XPack_int8 *InstancePtr);

void XPack_int8_Set_num_elements(XPack_int8 *InstancePtr, u32 Data);
u32 XPack_int8_Get_num_elements(XPack_int8 *InstancePtr);

void XPack_int8_InterruptGlobalEnable(XPack_int8 *InstancePtr);
void XPack_int8_InterruptGlobalDisable(XPack_int8 *InstancePtr);
void XPack_int8_InterruptEnable(XPack_int8 *InstancePtr, u32 Mask);
void XPack_int8_InterruptDisable(XPack_int8 *InstancePtr, u32 Mask);
void XPack_int8_InterruptClear(XPack_int8 *InstancePtr, u32 Mask);
u32 XPack_int8_InterruptGetEnabled(XPack_int8 *InstancePtr);
u32 XPack_int8_InterruptGetStatus(XPack_int8 *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
