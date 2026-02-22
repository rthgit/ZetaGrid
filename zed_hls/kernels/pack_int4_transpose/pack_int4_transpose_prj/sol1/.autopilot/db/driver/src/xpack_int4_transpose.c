// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xpack_int4_transpose.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XPack_int4_transpose_CfgInitialize(XPack_int4_transpose *InstancePtr, XPack_int4_transpose_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XPack_int4_transpose_Start(XPack_int4_transpose *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XPack_int4_transpose_ReadReg(InstancePtr->Control_BaseAddress, XPACK_INT4_TRANSPOSE_CONTROL_ADDR_AP_CTRL) & 0x80;
    XPack_int4_transpose_WriteReg(InstancePtr->Control_BaseAddress, XPACK_INT4_TRANSPOSE_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XPack_int4_transpose_IsDone(XPack_int4_transpose *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XPack_int4_transpose_ReadReg(InstancePtr->Control_BaseAddress, XPACK_INT4_TRANSPOSE_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XPack_int4_transpose_IsIdle(XPack_int4_transpose *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XPack_int4_transpose_ReadReg(InstancePtr->Control_BaseAddress, XPACK_INT4_TRANSPOSE_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XPack_int4_transpose_IsReady(XPack_int4_transpose *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XPack_int4_transpose_ReadReg(InstancePtr->Control_BaseAddress, XPACK_INT4_TRANSPOSE_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XPack_int4_transpose_EnableAutoRestart(XPack_int4_transpose *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XPack_int4_transpose_WriteReg(InstancePtr->Control_BaseAddress, XPACK_INT4_TRANSPOSE_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XPack_int4_transpose_DisableAutoRestart(XPack_int4_transpose *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XPack_int4_transpose_WriteReg(InstancePtr->Control_BaseAddress, XPACK_INT4_TRANSPOSE_CONTROL_ADDR_AP_CTRL, 0);
}

void XPack_int4_transpose_Set_num_elements(XPack_int4_transpose *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XPack_int4_transpose_WriteReg(InstancePtr->Control_BaseAddress, XPACK_INT4_TRANSPOSE_CONTROL_ADDR_NUM_ELEMENTS_DATA, Data);
}

u32 XPack_int4_transpose_Get_num_elements(XPack_int4_transpose *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XPack_int4_transpose_ReadReg(InstancePtr->Control_BaseAddress, XPACK_INT4_TRANSPOSE_CONTROL_ADDR_NUM_ELEMENTS_DATA);
    return Data;
}

void XPack_int4_transpose_InterruptGlobalEnable(XPack_int4_transpose *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XPack_int4_transpose_WriteReg(InstancePtr->Control_BaseAddress, XPACK_INT4_TRANSPOSE_CONTROL_ADDR_GIE, 1);
}

void XPack_int4_transpose_InterruptGlobalDisable(XPack_int4_transpose *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XPack_int4_transpose_WriteReg(InstancePtr->Control_BaseAddress, XPACK_INT4_TRANSPOSE_CONTROL_ADDR_GIE, 0);
}

void XPack_int4_transpose_InterruptEnable(XPack_int4_transpose *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XPack_int4_transpose_ReadReg(InstancePtr->Control_BaseAddress, XPACK_INT4_TRANSPOSE_CONTROL_ADDR_IER);
    XPack_int4_transpose_WriteReg(InstancePtr->Control_BaseAddress, XPACK_INT4_TRANSPOSE_CONTROL_ADDR_IER, Register | Mask);
}

void XPack_int4_transpose_InterruptDisable(XPack_int4_transpose *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XPack_int4_transpose_ReadReg(InstancePtr->Control_BaseAddress, XPACK_INT4_TRANSPOSE_CONTROL_ADDR_IER);
    XPack_int4_transpose_WriteReg(InstancePtr->Control_BaseAddress, XPACK_INT4_TRANSPOSE_CONTROL_ADDR_IER, Register & (~Mask));
}

void XPack_int4_transpose_InterruptClear(XPack_int4_transpose *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XPack_int4_transpose_WriteReg(InstancePtr->Control_BaseAddress, XPACK_INT4_TRANSPOSE_CONTROL_ADDR_ISR, Mask);
}

u32 XPack_int4_transpose_InterruptGetEnabled(XPack_int4_transpose *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XPack_int4_transpose_ReadReg(InstancePtr->Control_BaseAddress, XPACK_INT4_TRANSPOSE_CONTROL_ADDR_IER);
}

u32 XPack_int4_transpose_InterruptGetStatus(XPack_int4_transpose *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XPack_int4_transpose_ReadReg(InstancePtr->Control_BaseAddress, XPACK_INT4_TRANSPOSE_CONTROL_ADDR_ISR);
}

