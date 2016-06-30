#include <Python.h>

PyObject *slow_module = NULL;


PyObject *
slow(char *name, PyObject *args, PyObject *kwds)
{
    PyObject *func = NULL;
    PyObject *out = NULL;

    if (slow_module == NULL) {
        slow_module = PyImport_ImportModule("bottleneck.slow");
    }

    if (slow_module != NULL) {
        func = PyObject_GetAttrString(slow_module, name);
        if (func && PyCallable_Check(func)) {
            out = PyObject_Call(func, args, kwds);
            if (out == NULL) {
                Py_DECREF(func);
                return NULL;
            }
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
        }
        Py_XDECREF(func);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", name);
        return NULL;
    }

    return out;
}
