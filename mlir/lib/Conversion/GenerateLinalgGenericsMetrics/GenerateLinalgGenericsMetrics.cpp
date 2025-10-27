#include "mlir/IR/Builders.h"
// #include "mlir/IR/MLIRContext.h"
// #include "mlir/IR/Module.h"
// #include "mlir/IR/Block.h"
// #include "mlir/IR/BuiltinOps.h"
// #include "mlir/IR/Operation.h"
// #include "mlir/Pass/PassRegistry.h"
// #include "mlir/IR/Function.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

// #include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Conversion/GenerateLinalgGenericsMetrics/GenerateLinalgGenericsMetrics.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::linalg;

namespace mlir {

// / Pass 2: Analyze linalg.generic ops in the module and emit analytics
// / (number of loop dims, parallel vs reduction counts, reduction format).
struct GenerateLinalgGenericsMetricsPass
    : public PassWrapper<GenerateLinalgGenericsMetricsPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateLinalgGenericsMetricsPass)

  StringRef getArgument() const final {
    return "generate-linalg-generics-metrics";
  }

  StringRef getDescription() const final {
    return "Generates metrics of all linalg.generics operations in the input "
           "MLIR, "
           "and outputs the statistics in 'metrics.csv' file";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Open CSV file to write results
    std::string csvName = "metrics.csv";
    std::error_code ec;
    llvm::raw_fd_ostream csvOut(csvName, ec);
    if (ec) {
      llvm::errs() << "Failed to open " << csvName
                   << " for writing: " << ec.message() << "\n";
      return;
    }

    llvm::outs() << "This pass is running\n";
    csvOut << "op_index,op_name,func_name,num_loops,num_parallel,num_reduction,"
              "reduction_format,reduction_dims\n";
    llvm::outs() << "Initial Outputs happen\n";

    unsigned idx = 0;
    SmallVector<linalg::GenericOp, 8> ops;
    module.walk([&](linalg::GenericOp gop) { ops.push_back(gop); });

    llvm::outs() << "Initial walk successful\n";

    for (auto &gop : ops) {
      // Number of loop dimensions = number of iterator types if available
      //   SmallVector<StringRef, 4> iterTypes;
      SmallVector<linalg::IteratorTypeAttr, 4> iterTypes;

      //   if (auto iters = gop.iterator_types()) {
      //     for (auto attr : iters) {
      //       if (auto s = attr.dyn_cast<StringAttr>())
      //         iterTypes.push_back(s.getValue());
      //       else if (auto s = attr.dyn_cast<mlir::Attribute>())
      //         iterTypes.push_back(s.cast<StringAttr>().getValue());
      //     }
      //   } else if (auto attr =
      //   gop->getAttrOfType<ArrayAttr>("iterator_types")) {
      //     for (auto elem : attr) {
      //       iterTypes.push_back(elem.cast<StringAttr>().getValue());
      //     }
      //   }

      auto attr = gop->getAttrOfType<ArrayAttr>("iterator_types");
      if (attr)
        for (auto elem : attr) {
          if (auto iterTypeElem =
                  mlir::dyn_cast<linalg::IteratorTypeAttr>(elem)) {
            iterTypes.push_back(iterTypeElem);
          }
        }

      int numLoops = (int)iterTypes.size();
      int numParallel = 0;
      int numReduction = 0;
      SmallVector<int, 4> reductionDims;

      for (unsigned i = 0; i < iterTypes.size(); ++i) {
        // if (iterTypes[i] == "parallel")
        mlir::utils::IteratorType iteratorKind = iterTypes[i].getValue();
        switch (iteratorKind) {
        case mlir::utils::IteratorType::parallel:
          ++numParallel;
          break;
        case mlir::utils::IteratorType::reduction:
          ++numReduction;
          reductionDims.push_back(i);
          break;
          //   case mlir::utils::IteratorType::window:
          //     // Handle window dimension if necessary, otherwise ignore
          //     break;
        }
      }

      // Determine reduction format: outer / inner / mixed / none
      StringRef reductionFormat = "none";
      if (numReduction > 0) {
        bool allOuter = true;
        bool allInner = true;
        // define outer = lower index (closer to 0) and inner = higher index
        // we consider 'outer' if the reduction dims are among the first half
        // of dimensions and 'inner' if among the latter half. If mixed,
        // report mixed.
        unsigned half = numLoops / 2;
        for (int d : reductionDims) {
          if ((unsigned)d >= half)
            allOuter = false;
          if ((unsigned)d < half)
            allInner = false;
        }
        if (allOuter)
          reductionFormat = "outer";
        else if (allInner)
          reductionFormat = "inner";
        else
          reductionFormat = "mixed";
      }

      llvm::outs() << "Generated generation: " << idx << "\n";

      // Gather function name if available
      std::string funcName = "<unknown>";
      if (gop->getParentOfType<func::FuncOp>())
        funcName = gop->getParentOfType<func::FuncOp>().getName().str();

      llvm::outs() << "linalg.generic[" << idx << "]: func=" << funcName
                   << " loops=" << numLoops << " parallel=" << numParallel
                   << " reduction=" << numReduction
                   << " format=" << reductionFormat << "\n";

      // Write CSV row
      csvOut << idx << "," << gop->getName().getStringRef().str() << ","
             << funcName << "," << numLoops << "," << numParallel << ","
             << numReduction << "," << reductionFormat << ",\"";
      for (size_t i = 0; i < reductionDims.size(); ++i) {
        csvOut << reductionDims[i];
        if (i + 1 < reductionDims.size())
          csvOut << ";";
      }
      csvOut << "\"\n";

      ++idx;
    }

    llvm::outs() << "Wrote analytics to " << csvName << "\n";
  }
};

std::unique_ptr<Pass> createGenerateLinalgGenericsMetricsPass() {
  return std::make_unique<GenerateLinalgGenericsMetricsPass>();
}

} // namespace mlir
