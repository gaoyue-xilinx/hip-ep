// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-externalize-constants='externalize-min-num-elements=1 skip-constant-data=true' %s | FileCheck %s

// CHECK: hipdnn.constant_offsets = array<i64: 0, 0, 0, 0>
// CHECK: hipdnn.constant_source_kinds = array<i32: 4, 4, 4, 4>
// CHECK-COUNT-4: hip.external_data = {disk_resident

func.func @qmoe_tables(%ctx: !hip.context, %input: tensor<1x4xf16>,
                       %router: tensor<1x2xf16>, %output: tensor<1x4xf16>)
    -> tensor<1x4xf16> {
  %fc1w = hip.constant {location = "/models/text.onnx.data", offset = 0 : i64,
                        size = 16 : i64} : tensor<2x4x2xui8>
  %fc1s = hip.constant {location = "/models/text.onnx.data", offset = 16 : i64,
                        size = 16 : i64} : tensor<2x4x1xf16>
  %fc2w = hip.constant {location = "/models/text.onnx.data", offset = 32 : i64,
                        size = 8 : i64} : tensor<2x4x1xui8>
  %fc2s = hip.constant {location = "/models/text.onnx.data", offset = 40 : i64,
                        size = 16 : i64} : tensor<2x4x1xf16>
  %result = hip.qmoe(%ctx) ins(
      %input, %router, %fc1w, %fc1s, %fc2w, %fc2s :
      tensor<1x4xf16>, tensor<1x2xf16>, tensor<2x4x2xui8>,
      tensor<2x4x1xf16>, tensor<2x4x1xui8>, tensor<2x4x1xf16>)
      outs(%output : tensor<1x4xf16>)
      {expert_weight_bits = 4 : i64, k = 1 : i64, block_size = 4 : i64,
       normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64,
       use_sparse_mixer = 0 : i64, activation_alpha = 1.0 : f32,
       activation_beta = 0.0 : f32, swiglu_limit = 7.0 : f32,
       activation_type = "swiglu"} : tensor<1x4xf16>
  return %result : tensor<1x4xf16>
}
