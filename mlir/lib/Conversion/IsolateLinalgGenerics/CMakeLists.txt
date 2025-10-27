add_mlir_conversion_library(MLIRIsolateLinalgGenerics
  IsolateLinalgGenericsPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/IsolateLinalgGenerics

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRTransforms
  MLIRVectorDialect
  MLIRVectorTransforms
  )
