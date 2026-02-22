// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#ifdef SDT
#include "xparameters.h"
#endif
#include "xkv_page_ops.h"

extern XKv_page_ops_Config XKv_page_ops_ConfigTable[];

#ifdef SDT
XKv_page_ops_Config *XKv_page_ops_LookupConfig(UINTPTR BaseAddress) {
	XKv_page_ops_Config *ConfigPtr = NULL;

	int Index;

	for (Index = (u32)0x0; XKv_page_ops_ConfigTable[Index].Name != NULL; Index++) {
		if (!BaseAddress || XKv_page_ops_ConfigTable[Index].Control_BaseAddress == BaseAddress) {
			ConfigPtr = &XKv_page_ops_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XKv_page_ops_Initialize(XKv_page_ops *InstancePtr, UINTPTR BaseAddress) {
	XKv_page_ops_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XKv_page_ops_LookupConfig(BaseAddress);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XKv_page_ops_CfgInitialize(InstancePtr, ConfigPtr);
}
#else
XKv_page_ops_Config *XKv_page_ops_LookupConfig(u16 DeviceId) {
	XKv_page_ops_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XKV_PAGE_OPS_NUM_INSTANCES; Index++) {
		if (XKv_page_ops_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XKv_page_ops_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XKv_page_ops_Initialize(XKv_page_ops *InstancePtr, u16 DeviceId) {
	XKv_page_ops_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XKv_page_ops_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XKv_page_ops_CfgInitialize(InstancePtr, ConfigPtr);
}
#endif

#endif

