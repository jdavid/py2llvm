#include <llvm-c/Core.h>


LLVMBool llvm_init();
uint64_t llvm_compile(const char *filename, const char* fname);
void llvm_dispose();
