#include <Python.h>
#include "lib/prefilter.h"


PyObject *
py_llvm_init(PyObject *self, PyObject *args)
{
    llvm_init();
    Py_RETURN_NONE;
}

PyObject *
py_llvm_dispose(PyObject *self, PyObject *args)
{
    llvm_dispose();
    Py_RETURN_NONE;
}

PyObject *
py_llvm_compile(PyObject *self, PyObject *args)
{
    const char *filename, *fname;

    if (!PyArg_ParseTuple(args, "ss", &filename, &fname))
        return NULL;

    uint64_t address = llvm_compile(filename, fname);
    return PyLong_FromUnsignedLong(address);
}

static PyMethodDef methods[] = {
    {"llvm_init", py_llvm_init, METH_NOARGS, ""},
    {"llvm_dispose", py_llvm_dispose, METH_NOARGS, ""},
    {"llvm_compile", py_llvm_compile, METH_VARARGS, ""},
    {NULL}
};

static PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "prefilter",
    .m_doc = "",
    .m_size = -1,
    .m_methods = methods,
};

PyMODINIT_FUNC PyInit_prefilter(void)
{
    PyObject* module = PyModule_Create(&moduledef);
    if (module == NULL)
        return NULL;

    return module;
}
