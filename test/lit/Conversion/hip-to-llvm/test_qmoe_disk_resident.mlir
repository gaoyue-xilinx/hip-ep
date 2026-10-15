// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  func.func @test_qmoe_disk(%ctx: !hip.context,
      %input: memref<1x4xf16, 1>, %router: memref<1x2xf16, 1>,
      %output: memref<1x4xf16, 1>) {
    %c0 = arith.constant 10 : i64
    %c1 = arith.constant 11 : i64
    %c2 = arith.constant 12 : i64
    %c3 = arith.constant 13 : i64
    %fc1w = hip.get_constant(%ctx, %c0) {hip.disk_resident} : memref<2x4x2xui8, 1>
    %fc1s = hip.get_constant(%ctx, %c1) {hip.disk_resident} : memref<2x4x1xf16, 1>
    %fc2w = hip.get_constant(%ctx, %c2) {hip.disk_resident} : memref<2x4x1xui8, 1>
    %fc2s = hip.get_constant(%ctx, %c3) {hip.disk_resident} : memref<2x4x1xf16, 1>
    hip.qmoe(%ctx) ins(
        %input, %router, %fc1w, %fc1s, %fc2w, %fc2s :
        memref<1x4xf16, 1>, memref<1x2xf16, 1>, memref<2x4x2xui8, 1>,
        memref<2x4x1xf16, 1>, memref<2x4x1xui8, 1>, memref<2x4x1xf16, 1>)
        outs(%output : memref<1x4xf16, 1>)
        {expert_weight_bits = 4 : i64, k = 1 : i64, block_size = 4 : i64,
         normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64,
         use_sparse_mixer = 0 : i64, activation_alpha = 1.0 : f32,
         activation_beta = 0.0 : f32, swiglu_limit = 7.0 : f32,
         activation_type = "swiglu"}
    return
  }
}

// CHECK-LABEL: llvm.func @test_qmoe_disk
// CHECK-NOT: llvm.call @hipdnn_ep_constant_get
// CHECK: llvm.call @wrap_qmoe_from_file
// CHECK-SAME: (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64, !llvm.ptr
