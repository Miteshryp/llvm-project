#include "mlir/Conversion/IsolateLinalgGenerics/IsolateLinalgGenericsPass.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace mlir::linalg;

namespace mlir {

struct IsolateLinalgGenericsPass
    : public PassWrapper<IsolateLinalgGenericsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IsolateLinalgGenericsPass)

  StringRef getArgument() const final { return "isolate-linalg-generics"; }

  StringRef getDescription() const final {
    return "Isolate each linalg.generic into a separate MLIR file for "
           "inspection";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // index for naming files
    unsigned idx = 0;

    // Walk the module and collect linalg.generic ops

    SmallVector<linalg::GenericOp, 8> ops;
    module.walk([&](linalg::GenericOp lop) { ops.push_back(lop); });

    for (auto &gop : ops) {
      // Create an empty module
      OpBuilder builder(ctx);
      auto fileModule = ModuleOp::create(builder.getUnknownLoc());

      // Extracting Operators in the module
      llvm::SetVector<Value> externalOperands;

      // Collect SSA Arguments to the linalg.generic operation
      externalOperands.insert(gop->getOperands().begin(),
                              gop->getOperands().end());

      // Finding and storing SSA values used in the region of operation
      // Linalg.generics are guaranteed to have region, so region count guard is
      // not needed here
      Region &region = gop->getRegion(0);
      getUsedValuesDefinedAbove(region, externalOperands);

      // Extracting the type of all arguments needed in the linalg.generic op
      // (This includes the SSA arguments needed in the region of operation)
      SmallVector<Type> argTypes;
      for (Value val : externalOperands) {
        argTypes.push_back(val.getType());
      }

      // Extracting return type information from operation
      auto resTypes = gop->getResultTypes();

      // Create function signature with valid input and return types
      auto funcType = builder.getFunctionType(argTypes, resTypes);
      auto func = func::FuncOp::create(
          builder.getUnknownLoc(),
          (Twine("extracted_linalg_generic_") + Twine(idx)).str(), funcType);
      auto &entryBlock = *func.addEntryBlock();

      // Mapping extracted signatures to SSAs in the module
      IRMapping mapping;
      for (auto const &pair : llvm::enumerate(externalOperands)) {
        mapping.map(pair.value(), entryBlock.getArgument(pair.index()));
      }

      // Clone op into the new function's block
      OpBuilder funcBuilder(&entryBlock, entryBlock.begin());
      Operation *cloned = funcBuilder.clone(*gop.getOperation(), mapping);

      // Create function with valid argument and return type signatures, as well
      // as input SSAs mapped as arguments to the function containing the op
      funcBuilder.create<func::ReturnOp>(builder.getUnknownLoc(),
                                         cloned->getResults());

      fileModule.push_back(func);

      // Print the module to a file
      std::string filename =
          (Twine("linalg_generic_") + Twine(idx) + ".mlir").str();
      std::error_code ec;
      llvm::raw_fd_ostream out(filename, ec);
      if (ec) {
        llvm::errs() << "Failed to open " << filename
                     << " for writing: " << ec.message() << "\n";
      } else {
        fileModule.print(out);
        llvm::outs() << "Wrote " << filename << "\n";
      }

      ++idx;
    }
  }
};

std::unique_ptr<Pass> createIsolateLinalgGenericsPass() {
  return std::make_unique<IsolateLinalgGenericsPass>();
}

} // namespace mlir
