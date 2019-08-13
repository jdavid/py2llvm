/**
 * Load a LLVM IR file and execute it.
 *
 * The signature of the IR function should be prefilter from Blosc2
 * int (*blosc2_prefilter_fn)(blosc2_prefilter_params* params);
 *
 * The LLVM part is originally based in
 * https://github.com/owst/getting-started-with-the-newer-llvm-c-api/blob/master/sum.c
 *
 * LLVM C API reference
 * https://llvm.org/doxygen/group__LLVMC.html
 */

//#include <inttypes.h>
#include <stdio.h>
//#include <stdlib.h>

//#include <llvm-c/Core.h>
#include <llvm-c/ExecutionEngine.h>
//#include <llvm-c/Target.h>
#include <llvm-c/Analysis.h>
#include <llvm-c/IRReader.h>
#include <llvm-c/Transforms/PassManagerBuilder.h>

#include "prefilter.h"
//#include "c-blosc2/blosc/blosc.h"


// Globals
LLVMContextRef context;
LLVMExecutionEngineRef engine;


/*
 * We optimize the whole module, so we don't need to optimize
 * function-by-function. We set -O3 so we don't have to specify optimization
 * passes.
 */
int optimize(LLVMExecutionEngineRef *engine, LLVMModuleRef *mod) {
    LLVMPassManagerBuilderRef pmb = LLVMPassManagerBuilderCreate();
    LLVMPassManagerBuilderSetOptLevel(pmb, 3); // Opt level 0-3
    //LLVMPassManagerBuilderSetSizeLevel(pmb, 2); // Size level 0-2

    // function-by-function pass pipeline
/*
    LLVMValueRef fref;
    LLVMFindFunction(*engine, "poly", &fref);
    //LLVMFindFunction(*engine, fname, &fref);
    LLVMPassManagerRef fpm = LLVMCreateFunctionPassManagerForModule(*mod);
    LLVMPassManagerBuilderPopulateFunctionPassManager(pmb, fpm);
    LLVMInitializeFunctionPassManager(fpm);
    LLVMRunFunctionPassManager(fpm, fref);
    LLVMFinalizeFunctionPassManager(fpm);
    LLVMDisposePassManager(fpm);
*/

    // whole-module pass pipeline
    LLVMPassManagerRef mpm = LLVMCreatePassManager();
    LLVMPassManagerBuilderPopulateModulePassManager(pmb, mpm);
//  LLVMAddTargetData(LLVMGetExecutionEngineTargetData(engine), mpm);
//  LLVMAddConstantPropagationPass(mpm);
//  LLVMAddInstructionCombiningPass(mpm);
//  LLVMAddPromoteMemoryToRegisterPass(mpm);
//  LLVMAddGVNPass(mpm);
//  LLVMAddCFGSimplificationPass(mpm);
    LLVMRunPassManager(mpm, *mod);
    LLVMDisposePassManager(mpm);

    LLVMPassManagerBuilderDispose(pmb);

    return 0;
}


LLVMBool llvm_init()
{
    LLVMBool error;

    //LLVMLinkInMCJIT();
    error = LLVMInitializeNativeTarget();
    error = LLVMInitializeNativeAsmPrinter();

    return error;
}

void llvm_dispose()
{
    LLVMDisposeExecutionEngine(engine);
    LLVMContextDispose(context);

    // for some strange reason, this does a "pointer being freed was not allocated"
    //LLVMDisposeMemoryBuffer(memoryBuffer);
}


uint64_t llvm_compile(const char *filename, const char *fname)
{
    uint64_t address = 0;

    // Output message returned by LLVM. Disposed with LLVMDisposeMessage (same
    // as free, per implementation). May return a message even on success
    // (found it once returning an empty string...)
    char *message = NULL;
    LLVMBool error;

    // Read the IR file into a buffer
    LLVMMemoryBufferRef buffer;
    error = LLVMCreateMemoryBufferWithContentsOfFile(filename, &buffer, &message);
    if (error)
    {
        fprintf(stderr, "%s\n", message);
        goto exit;
    }

    // Parse IR in memory buffer, creates module
    LLVMModuleRef mod;
    context = LLVMContextCreate();
    error = LLVMParseIRInContext(context, buffer, &mod, &message);
    if (error)
    {
        fprintf(stderr, "Invalid IR detected! message: '%s'\n", message);
        LLVMDisposeMemoryBuffer(buffer);
        goto exit;
    }

    error = LLVMVerifyModule(mod, LLVMAbortProcessAction, &message);
    if (error)
    {
        fprintf(stderr, "IR not verified! error: '%s'\n", message);
        LLVMDisposeMemoryBuffer(buffer);
        goto exit;
    }

    // Create execution engine for module
    error = LLVMCreateExecutionEngineForModule(&engine, mod, &message);
    if (error)
    {
        fprintf(stderr, "failed to create execution engine: '%s'\n", message);
        goto exit;
    }

    //
    // Optimize
    //

    optimize(&engine, &mod);
    // Output optimized IR for verification
    //LLVMPrintModuleToFile(mod, "debug_O3.ll", &message);

    // Function address
    address = LLVMGetFunctionAddress(engine, fname);

exit:
    LLVMDisposeMessage(message);
    return address;
}
