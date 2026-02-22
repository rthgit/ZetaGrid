// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef XKV_PAGE_OPS_H
#define XKV_PAGE_OPS_H

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
#include "xkv_page_ops_hw.h"

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
} XKv_page_ops_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XKv_page_ops;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XKv_page_ops_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XKv_page_ops_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XKv_page_ops_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XKv_page_ops_ReadReg(BaseAddress, RegOffset) \
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
int XKv_page_ops_Initialize(XKv_page_ops *InstancePtr, UINTPTR BaseAddress);
XKv_page_ops_Config* XKv_page_ops_LookupConfig(UINTPTR BaseAddress);
#else
int XKv_page_ops_Initialize(XKv_page_ops *InstancePtr, u16 DeviceId);
XKv_page_ops_Config* XKv_page_ops_LookupConfig(u16 DeviceId);
#endif
int XKv_page_ops_CfgInitialize(XKv_page_ops *InstancePtr, XKv_page_ops_Config *ConfigPtr);
#else
int XKv_page_ops_Initialize(XKv_page_ops *InstancePtr, const char* InstanceName);
int XKv_page_ops_Release(XKv_page_ops *InstancePtr);
#endif

void XKv_page_ops_Start(XKv_page_ops *InstancePtr);
u32 XKv_page_ops_IsDone(XKv_page_ops *InstancePtr);
u32 XKv_page_ops_IsIdle(XKv_page_ops *InstancePtr);
u32 XKv_page_ops_IsReady(XKv_page_ops *InstancePtr);
void XKv_page_ops_EnableAutoRestart(XKv_page_ops *InstancePtr);
void XKv_page_ops_DisableAutoRestart(XKv_page_ops *InstancePtr);

void XKv_page_ops_Set_num_cmds(XKv_page_ops *InstancePtr, u32 Data);
u32 XKv_page_ops_Get_num_cmds(XKv_page_ops *InstancePtr);

void XKv_page_ops_InterruptGlobalEnable(XKv_page_ops *InstancePtr);
void XKv_page_ops_InterruptGlobalDisable(XKv_page_ops *InstancePtr);
void XKv_page_ops_InterruptEnable(XKv_page_ops *InstancePtr, u32 Mask);
void XKv_page_ops_InterruptDisable(XKv_page_ops *InstancePtr, u32 Mask);
void XKv_page_ops_InterruptClear(XKv_page_ops *InstancePtr, u32 Mask);
u32 XKv_page_ops_InterruptGetEnabled(XKv_page_ops *InstancePtr);
u32 XKv_page_ops_InterruptGetStatus(XKv_page_ops *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
