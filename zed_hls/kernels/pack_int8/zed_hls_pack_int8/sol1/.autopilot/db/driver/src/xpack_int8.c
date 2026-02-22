// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xpack_int8.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XPack_int8_CfgInitialize(XPack_int8 *InstancePtr, XPack_int8_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XPack_int8_Start(XPack_int8 *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XPack_int8_ReadReg(InstancePtr->Control_BaseAddress, XPACK_INT8_CONTROL_ADDR_AP_CTRL) & 0x80;
    XPack_int8_WriteReg(InstancePtr->Control_BaseAddress, XPACK_INT8_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XPack_int8_IsDone(XPack_int8 *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XPack_int8_ReadReg(InstancePtr->Control_BaseAddress, XPACK_INT8_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XPack_int8_IsIdle(XPack_int8 *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XPack_int8_ReadReg(InstancePtr->Control_BaseAddress, XPACK_INT8_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XPack_int8_IsReady(XPack_int8 *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XPack_int8_ReadReg(InstancePtr->Control_BaseAddress, XPACK_INT8_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XPack_int8_EnableAutoRestart(XPack_int8 *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XPack_int8_WriteReg(InstancePtr->Control_BaseAddress, XPACK_INT8_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XPack_int8_DisableAutoRestart(XPack_int8 *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XPack_int8_WriteReg(InstancePtr->Control_BaseAddress, XPACK_INT8_CONTROL_ADDR_AP_CTRL, 0);
}

void XPack_int8_Set_num_elements(XPack_int8 *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XPack_int8_WriteReg(InstancePtr->Control_BaseAddress, XPACK_INT8_CONTROL_ADDR_NUM_ELEMENTS_DATA, Data);
}

u32 XPack_int8_Get_num_elements(XPack_int8 *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XPack_int8_ReadReg(InstancePtr->Control_BaseAddress, XPACK_INT8_CONTROL_ADDR_NUM_ELEMENTS_DATA);
    return Data;
}

void XPack_int8_InterruptGlobalEnable(XPack_int8 *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XPack_int8_WriteReg(InstancePtr->Control_BaseAddress, XPACK_INT8_CONTROL_ADDR_GIE, 1);
}

void XPack_int8_InterruptGlobalDisable(XPack_int8 *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XPack_int8_WriteReg(InstancePtr->Control_BaseAddress, XPACK_INT8_CONTROL_ADDR_GIE, 0);
}

void XPack_int8_InterruptEnable(XPack_int8 *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XPack_int8_ReadReg(InstancePtr->Control_BaseAddress, XPACK_INT8_CONTROL_ADDR_IER);
    XPack_int8_WriteReg(InstancePtr->Control_BaseAddress, XPACK_INT8_CONTROL_ADDR_IER, Register | Mask);
}

void XPack_int8_InterruptDisable(XPack_int8 *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XPack_int8_ReadReg(InstancePtr->Control_BaseAddress, XPACK_INT8_CONTROL_ADDR_IER);
    XPack_int8_WriteReg(InstancePtr->Control_BaseAddress, XPACK_INT8_CONTROL_ADDR_IER, Register & (~Mask));
}

void XPack_int8_InterruptClear(XPack_int8 *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XPack_int8_WriteReg(InstancePtr->Control_BaseAddress, XPACK_INT8_CONTROL_ADDR_ISR, Mask);
}

u32 XPack_int8_InterruptGetEnabled(XPack_int8 *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XPack_int8_ReadReg(InstancePtr->Control_BaseAddress, XPACK_INT8_CONTROL_ADDR_IER);
}

u32 XPack_int8_InterruptGetStatus(XPack_int8 *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XPack_int8_ReadReg(InstancePtr->Control_BaseAddress, XPACK_INT8_CONTROL_ADDR_ISR);
}

