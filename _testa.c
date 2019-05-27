/*
 * It's possible to call a function if you know the memory address
 * (dynamically) and the prototype (statically). For example:
 *
 *   int x = (int (*)(void))ptr();
 *
 * If you only know the prototype dynamically, it's still possible to call the
 * function using libffi:
 *
 *   https://sourceware.org/libffi/
 *
 * Actually ctypes uses libffi.
 */

#include <Python.h>


PyObject* run(PyObject* self, PyObject* args)
{
    PyObject* function;
    PyObject* arguments;

    if (!PyArg_ParseTuple(args, "OO!", &function, &arguments, &PyTuple_Type))
    {
        return NULL;
    }

    // .cfunction must be instance of _ctypes.PyCFuncPtr
    PyObject* cfunction = PyObject_GetAttrString(function, "cfunction");
    if (cfunction == NULL)
    {
        return NULL;
    }

    // Call
    PyObject* res = PyObject_CallObject(cfunction, arguments);
    Py_DECREF(cfunction);

    // Ok
    return res;
}

PyMethodDef m_methods[] = {
    {"run", run, METH_VARARGS, ""},
    {NULL}
};


// Module Definition struct
struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_testa",                        /* m_name */
    "Test Array",                    /* m_doc */
    -1,                              /* m_size */
    m_methods,                       /* m_methods */
    NULL,                            /* m_reload */
    NULL,                            /* m_traverse */
    NULL,                            /* m_clear */
    NULL,                            /* m_free */
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit__testa(void)
{
    return PyModule_Create(&moduledef);
}
