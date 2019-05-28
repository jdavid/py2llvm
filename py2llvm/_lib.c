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


typedef struct {
    PyObject_HEAD
    void* array;
    Py_ssize_t size;
} Array;

static PyObject* Array_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Array *self;
    self = (Array*)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->array = NULL;
        self->size = 0;
    }
    return (PyObject*)self;
}

static void Array_dealloc(Array* self)
{
    free(self->array);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int Array_init(Array *self, PyObject *args, PyObject *kwds)
{
    PyObject* list = NULL;
    PyObject* item = NULL;
    double value;

    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &list))
        return -1;

    self->size = PyList_Size(list);
    self->array = malloc(self->size * sizeof(double));

    for (Py_ssize_t i=0; i < self->size; i++)
    {
        item = PyList_GetItem(list, i);
        value = PyFloat_AsDouble(item);
        if (value == -1.0 && PyErr_Occurred())
            return -1;
    }

    return 0;
}


static PyTypeObject ArrayType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_lib.Array",
    .tp_doc = "",
    .tp_basicsize = sizeof(Array),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Array_new,
    .tp_dealloc = (destructor) Array_dealloc,
    .tp_init = (initproc) Array_init,

//  .tp_members = Array_members,
//  .tp_methods = Array_methods,
};


PyObject* run(PyObject* self, PyObject* args)
{
    PyObject* function;
    PyObject* arguments;

    if (!PyArg_ParseTuple(args, "OO!", &function, &PyTuple_Type, &arguments))
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
static PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_lib",
    .m_doc = "",
    .m_size = -1,
    .m_methods = m_methods,
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit__lib(void)
{
    if (PyType_Ready(&ArrayType) < 0)
        return NULL;

    PyObject* module = PyModule_Create(&moduledef);
    if (module == NULL)
        return NULL;

    Py_INCREF(&ArrayType);
    PyModule_AddObject(module, "Array", (PyObject*)&ArrayType);

    return module;
}
