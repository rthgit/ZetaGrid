// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xkv_page_ops.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XKv_page_ops_CfgInitialize(XKv_page_ops *InstancePtr, XKv_page_ops_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XKv_page_ops_Start(XKv_page_ops *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XKv_page_ops_ReadReg(InstancePtr->Control_BaseAddress, XKV_PAGE_OPS_CONTROL_ADDR_AP_CTRL) & 0x80;
    XKv_page_ops_WriteReg(InstancePtr->Control_BaseAddress, XKV_PAGE_OPS_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XKv_page_ops_IsDone(XKv_page_ops *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XKv_page_ops_ReadReg(InstancePtr->Control_BaseAddress, XKV_PAGE_OPS_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XKv_page_ops_IsIdle(XKv_page_ops *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XKv_page_ops_ReadReg(InstancePtr->Control_BaseAddress, XKV_PAGE_OPS_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XKv_page_ops_IsReady(XKv_page_ops *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XKv_page_ops_ReadReg(InstancePtr->Control_BaseAddress, XKV_PAGE_OPS_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XKv_page_ops_EnableAutoRestart(XKv_page_ops *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XKv_page_ops_WriteReg(InstancePtr->Control_BaseAddress, XKV_PAGE_OPS_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XKv_page_ops_DisableAutoRestart(XKv_page_ops *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XKv_page_ops_WriteReg(InstancePtr->Control_BaseAddress, XKV_PAGE_OPS_CONTROL_ADDR_AP_CTRL, 0);
}

void XKv_page_ops_Set_num_cmds(XKv_page_ops *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XKv_page_ops_WriteReg(InstancePtr->Control_BaseAddress, XKV_PAGE_OPS_CONTROL_ADDR_NUM_CMDS_DATA, Data);
}

u32 XKv_page_ops_Get_num_cmds(XKv_page_ops *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XKv_page_ops_ReadReg(InstancePtr->Control_BaseAddress, XKV_PAGE_OPS_CONTROL_ADDR_NUM_CMDS_DATA);
    return Data;
}

void XKv_page_ops_InterruptGlobalEnable(XKv_page_ops *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XKv_page_ops_WriteReg(InstancePtr->Control_BaseAddress, XKV_PAGE_OPS_CONTROL_ADDR_GIE, 1);
}

void XKv_page_ops_InterruptGlobalDisable(XKv_page_ops *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XKv_page_ops_WriteReg(InstancePtr->Control_BaseAddress, XKV_PAGE_OPS_CONTROL_ADDR_GIE, 0);
}

void XKv_page_ops_InterruptEnable(XKv_page_ops *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XKv_page_ops_ReadReg(InstancePtr->Control_BaseAddress, XKV_PAGE_OPS_CONTROL_ADDR_IER);
    XKv_page_ops_WriteReg(InstancePtr->Control_BaseAddress, XKV_PAGE_OPS_CONTROL_ADDR_IER, Register | Mask);
}

void XKv_page_ops_InterruptDisable(XKv_page_ops *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XKv_page_ops_ReadReg(InstancePtr->Control_BaseAddress, XKV_PAGE_OPS_CONTROL_ADDR_IER);
    XKv_page_ops_WriteReg(InstancePtr->Control_BaseAddress, XKV_PAGE_OPS_CONTROL_ADDR_IER, Register & (~Mask));
}

void XKv_page_ops_InterruptClear(XKv_page_ops *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XKv_page_ops_WriteReg(InstancePtr->Control_BaseAddress, XKV_PAGE_OPS_CONTROL_ADDR_ISR, Mask);
}

u32 XKv_page_ops_InterruptGetEnabled(XKv_page_ops *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XKv_page_ops_ReadReg(InstancePtr->Control_BaseAddress, XKV_PAGE_OPS_CONTROL_ADDR_IER);
}

u32 XKv_page_ops_InterruptGetStatus(XKv_page_ops *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XKv_page_ops_ReadReg(InstancePtr->Control_BaseAddress, XKV_PAGE_OPS_CONTROL_ADDR_ISR);
}

