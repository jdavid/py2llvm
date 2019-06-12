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
#include <alloca.h>


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
 * Function
 */

typedef struct {
    PyObject_HEAD
    ffi_cif cif; // libffi function definition
    void* fn; // pointer to function
    ffi_type** arg_types; // have to alloc this
} Function;

static PyObject* Function_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    Function *self;
    self = (Function*)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->arg_types = NULL;
    }
    return (PyObject*)self;
}

static void Function_dealloc(Function* self)
{
    free(self->arg_types);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

ffi_type* get_ffi_type(PyObject* object)
{
   if (object == NULL)
       return NULL;

   const char* argtype = PyUnicode_AsUTF8(object);
   if (argtype == NULL)
       return NULL;

   if (strcmp(argtype, "") == 0)
       return &ffi_type_void;

   if (strcmp(argtype, "p") == 0)
       return &ffi_type_pointer;

   if (strcmp(argtype, "i64") == 0)
       return &ffi_type_sint64;

   if (strcmp(argtype, "f64") == 0)
       return &ffi_type_double;

   PyErr_Format(PyExc_RuntimeError, "unexpected type %s", argtype);
   return NULL;
}


static PyObject* Function_prepare(Function* self, PyObject* function)
{
    PyObject* object;
    ffi_status status;
    ffi_abi abi = FFI_DEFAULT_ABI;

    // number of arguments
    if ((object = PyObject_GetAttrString(function, "nargs")) == NULL)
        return NULL;

    unsigned long nargs = PyLong_AsUnsignedLong(object);
    Py_DECREF(object);
    if (nargs == (unsigned long)-1 && PyErr_Occurred())
        return NULL;

    // argument types
    if (nargs == 0)
    {
        self->arg_types = NULL;
    }
    else
    {
        if ((object = PyObject_GetAttrString(function, "argtypes")) == NULL)
            return NULL;

        self->arg_types = malloc(nargs * sizeof(ffi_type*));
        for (unsigned long i=0; i < nargs; i++)
        {
            self->arg_types[i] = get_ffi_type(PyList_GetItem(object, (Py_ssize_t)i));
            if (self->arg_types[i] == NULL)
                return NULL;
        }
        Py_DECREF(object);
    }

    // return type
    if ((object = PyObject_GetAttrString(function, "rtype")) == NULL)
        return NULL;

    ffi_type* rtype = get_ffi_type(object);
    Py_DECREF(object);
    if (rtype == NULL)
        return NULL;

    // prepare
    status = ffi_prep_cif(&self->cif, abi, (unsigned int)nargs, rtype, self->arg_types);
    if (status != FFI_OK)
    {
        PyErr_Format(PyExc_RuntimeError, "ffi_prep_cif returned %d", status);
        return NULL;
    }

    // function address
    if ((object = PyObject_GetAttrString(function, "cfunction_ptr")) == NULL)
        return NULL;

    self->fn = PyLong_AsVoidPtr(object);
    Py_DECREF(object);
    if (self->fn == NULL && PyErr_Occurred())
        return NULL;

    Py_RETURN_NONE;
}


PyObject* Function_call(PyObject *obj, PyObject *args, PyObject *kwargs)
{
    PyObject* arguments;
    PyObject* object;

    Function* self = (Function*)obj;

    if (!PyArg_ParseTuple(args, "O!", &PyTuple_Type, &arguments))
        return NULL;

    // arguments
    void* avalues[self->cif.nargs];
    for (unsigned long i=0; i < self->cif.nargs; i++)
    {
        avalues[i] = alloca(self->cif.arg_types[i]->size);
        object = PyTuple_GetItem(arguments, (Py_ssize_t)i);
        if (self->cif.arg_types[i] == &ffi_type_pointer)
        {
           void* ptr = PyLong_AsVoidPtr(object);
           if (ptr == NULL && PyErr_Occurred()) { }
           *(void**)(avalues[i]) = ptr;
        }
        else if (self->cif.arg_types[i] == &ffi_type_sint64)
        {
           long value = PyLong_AsLong(object);
           if (value == -1 && PyErr_Occurred()) { }
           *(long*)(avalues[i]) = value;
        }
        else if (self->cif.arg_types[i] == &ffi_type_double)
        {
           double value = PyFloat_AsDouble(object);
           if (value == -1.0 && PyErr_Occurred()) { }
           *(double*)(avalues[i]) = value;
        }
    }

    // call
    void* rvalue = NULL;
    if (self->cif.rtype != &ffi_type_void)
        rvalue = alloca(self->cif.rtype->size);

    ffi_call(&self->cif, self->fn, rvalue, avalues);

    // return value
    PyObject* res = NULL;
    if (self->cif.rtype == &ffi_type_void)
    {
       res = Py_None;
       Py_INCREF(res);
    }
    else if (self->cif.rtype == &ffi_type_sint64)
    {
       res = PyLong_FromLong(*(long*)rvalue);
    }
    else if (self->cif.rtype == &ffi_type_double)
    {
       res = PyFloat_FromDouble(*(double*)rvalue);
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "unexpected return type");
    }

    // Ok
    return res;
}

PyMethodDef Function_methods[] = {
    {"prepare", (PyCFunction) Function_prepare, METH_O,
     "prepare the function to be called"},
    {NULL}
};

static PyTypeObject FunctionType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_lib.Function",
    .tp_doc = "",
    .tp_basicsize = sizeof(Function),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Function_new,
    .tp_dealloc = (destructor) Function_dealloc,
    .tp_call = (ternaryfunc) Function_call,
    .tp_methods = Function_methods,
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

    if (PyType_Ready(&FunctionType) < 0)
        return NULL;

    PyObject* module = PyModule_Create(&moduledef);
    if (module == NULL)
        return NULL;

    Py_INCREF(&ArrayType);
    PyModule_AddObject(module, "Array", (PyObject*)&ArrayType);

    Py_INCREF(&FunctionType);
    PyModule_AddObject(module, "Function", (PyObject*)&FunctionType);

    return module;
}
