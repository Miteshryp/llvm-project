add_mlir_conversion_library(MLIRGenerateLinalgGenericsMetrics
  GenerateLinalgGenericsMetrics.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/GenerateLinalgGenericsMetrics

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRTransforms
  MLIRVectorDialect
  MLIRVectorTransforms
  )
