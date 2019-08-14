#include <llvm-c/Core.h>


LLVMBool llvm_init();
uint64_t llvm_compile_file(const char *filename, const char* fname);
uint64_t llvm_compile_str(const char *data, const char* fname);
void llvm_dispose();
