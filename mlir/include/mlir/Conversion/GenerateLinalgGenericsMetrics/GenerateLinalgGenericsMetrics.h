//===- IsolateLinalgGenerics.h -  ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_GENERATE_LINALG_GENERICS_METRICS_H
#define MLIR_CONVERSION_GENERATE_LINALG_GENERICS_METRICS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
// #include "mlir/IR/Module.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassRegistry.h"
// #include "mlir/IR/Function.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <memory>




namespace mlir {

#define GEN_PASS_DECL_GENERATELINALGGENERICSMETRICSPASS
#include "mlir/Conversion/Passes.h.inc"


std::unique_ptr<Pass> createGenerateLinalgGenericsMetricsPass();


} // namespace ub
#endif // MLIR_CONVERSION_UBTOSPIRV_UBSPIRV_H
