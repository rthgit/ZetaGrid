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
#include "xpack_int8.h"

extern XPack_int8_Config XPack_int8_ConfigTable[];

#ifdef SDT
XPack_int8_Config *XPack_int8_LookupConfig(UINTPTR BaseAddress) {
	XPack_int8_Config *ConfigPtr = NULL;

	int Index;

	for (Index = (u32)0x0; XPack_int8_ConfigTable[Index].Name != NULL; Index++) {
		if (!BaseAddress || XPack_int8_ConfigTable[Index].Control_BaseAddress == BaseAddress) {
			ConfigPtr = &XPack_int8_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XPack_int8_Initialize(XPack_int8 *InstancePtr, UINTPTR BaseAddress) {
	XPack_int8_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XPack_int8_LookupConfig(BaseAddress);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XPack_int8_CfgInitialize(InstancePtr, ConfigPtr);
}
#else
XPack_int8_Config *XPack_int8_LookupConfig(u16 DeviceId) {
	XPack_int8_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XPACK_INT8_NUM_INSTANCES; Index++) {
		if (XPack_int8_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XPack_int8_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XPack_int8_Initialize(XPack_int8 *InstancePtr, u16 DeviceId) {
	XPack_int8_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XPack_int8_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XPack_int8_CfgInitialize(InstancePtr, ConfigPtr);
}
#endif

#endif

