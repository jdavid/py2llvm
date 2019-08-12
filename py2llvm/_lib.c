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
#include <alloca.h>
#include <math.h>


/*
 * C functions, to be called from LLVM, for testing purposes.
 */

double fun(double x)
{
    //printf("I'm funny\n");
    return sin(x);
}


/*
 * Array
 */

typedef struct {
    PyObject_HEAD
    void* data;
    Py_ssize_t size;
} Array;

static PyObject* Array_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    Array *self;
    self = (Array*)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->data = NULL;
        self->size = 0;
    }
    return (PyObject*)self;
}

static void Array_dealloc(Array* self)
{
    free(self->data);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int Array_init(Array *self, PyObject *args, PyObject *kwargs)
{
    PyObject* list = NULL;
    PyObject* item = NULL;
    double value;

    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &list))
        return -1;

    self->size = PyList_Size(list);
    self->data = malloc(self->size * sizeof(double));

    for (Py_ssize_t i=0; i < self->size; i++)
    {
        item = PyList_GetItem(list, i);
        value = PyFloat_AsDouble(item);
        if (value == -1.0 && PyErr_Occurred())
            return -1;

        ((double*)(self->data))[i] = value;
    }

    return 0;
}

static PyObject* Array_addr(Array* self, void* closure)
{
    return PyLong_FromVoidPtr(self->data);
}

static PyObject* Array_size(Array* self, void* closure)
{
    return PyLong_FromSsize_t(self->size);
}

static PyObject* Array_data(Array* self, void* closure)
{
    return PyMemoryView_FromMemory((char*)self->data, self->size * sizeof(double), PyBUF_WRITE);
}


static PyGetSetDef Array_getset[] = {
    {"addr", (getter)Array_addr, NULL, "address", NULL},
    {"size", (getter)Array_size, NULL, "size", NULL},
    {"data", (getter)Array_data, NULL, "data", NULL},
    {NULL}
};


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
    .tp_getset = Array_getset,
};


/*
 * Module
 */

// Module Definition struct
static PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_lib",
    .m_doc = "",
    .m_size = -1,
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
