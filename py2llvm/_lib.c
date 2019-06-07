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
#include <ffi.h>


typedef struct {
    PyObject_HEAD
    void* data;
    Py_ssize_t size;
} Array;

static PyObject* Array_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
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

static int Array_init(Array *self, PyObject *args, PyObject *kwds)
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


ffi_type* get_ffi_type(PyObject* object)
{
   if (object == NULL)
       return NULL;

   char* argtype = PyUnicode_AsUTF8(object);
   if (argtype == NULL)
       return NULL;

   if (strcmp(argtype, "p") == 0)
       return &ffi_type_pointer;

   if (strcmp(argtype, "i64") == 0)
       return &ffi_type_sint64;

   if (strcmp(argtype, "f64") == 0)
       return &ffi_type_double;

   PyErr_Format(PyExc_RuntimeError, "unexpected type %s", argtype);
   return NULL;
}

PyObject* run(PyObject* self, PyObject* args)
{
    PyObject* function;
    PyObject* arguments;
    PyObject* object;
    PyObject* res = NULL;

    if (!PyArg_ParseTuple(args, "OO!", &function, &PyTuple_Type, &arguments))
    {
        return NULL;
    }

    // libffi: prepare ffi_cif object
    ffi_cif cif;
    ffi_status status;
    ffi_abi abi = FFI_DEFAULT_ABI;

    // number of arguments
    unsigned long nargs;
    if ((object = PyObject_GetAttrString(function, "nargs")) == NULL)
        return NULL;

    nargs = PyLong_AsUnsignedLong(object);
    Py_DECREF(object);
    if (nargs == -1 && PyErr_Occurred()) { return NULL; }

    // argument types
    if ((object = PyObject_GetAttrString(function, "argtypes")) == NULL)
        return NULL;

    ffi_type* argtypes[nargs];
    for (unsigned long i=0; i < nargs; i++)
    {
        argtypes[i] = get_ffi_type(PyList_GetItem(object, (Py_ssize_t)i));
        if (argtypes[i] == NULL)
        {
            return NULL;
        }
    }
    Py_DECREF(object);

    // return type
    if ((object = PyObject_GetAttrString(function, "rtype")) == NULL)
        return NULL;

    ffi_type* rtype = get_ffi_type(object);
    Py_DECREF(object);

    // prepare
    status = ffi_prep_cif(&cif, abi, (unsigned int) nargs, rtype, argtypes);
    if (status != FFI_OK)
    {
        PyErr_Format(PyExc_RuntimeError, "ffi_prep_cif returned %d", status);
        return NULL;
    }

    // libffi: call
    // function address
    if ((object = PyObject_GetAttrString(function, "cfunction_ptr")) == NULL)
        return NULL;

    void* fn = PyLong_AsVoidPtr(object);
    Py_DECREF(object);
    if (fn == NULL && PyErr_Occurred()) { return NULL; }  // Error check

    // return value
    void* rvalue = malloc(rtype->size);

    // arguments
    void* avalues[nargs];
    for (unsigned long i=0; i < nargs; i++)
    {
        object = PyTuple_GetItem(arguments, (Py_ssize_t)i);
        if (argtypes[i] == &ffi_type_pointer)
        {
           void* ptr = PyLong_AsVoidPtr(object);
           if (ptr == NULL && PyErr_Occurred()) { }
           avalues[i] = &ptr;
        }
        else if (argtypes[i] == &ffi_type_sint64)
        {
           long volatile value = PyLong_AsLong(object);
           if (value == -1 && PyErr_Occurred()) { }
           avalues[i] = (void*)&value;
        }
        else if (argtypes[i] == &ffi_type_double)
        {
           double volatile value = PyFloat_AsDouble(object);
           if (value == -1.0 && PyErr_Occurred()) { }
           avalues[i] = (void*)&value;
        }
    }

    ffi_call(&cif, fn, rvalue, avalues);

    if (rtype == &ffi_type_double)
    {
       res = PyFloat_FromDouble(*(double*)rvalue);
    }

    // Free
    free(rvalue);

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
