#include <Python.h>


PyObject *
slow(char *name, PyObject *args, PyObject *kwds)
{
    PyObject *module = NULL;
    PyObject *func = NULL;
    PyObject *out = NULL;

    module = PyImport_ImportModule("bottleneck.slow");

    if (module != NULL) {
        func = PyObject_GetAttrString(module, name);
        if (func && PyCallable_Check(func)) {
            out = PyObject_Call(func, args, kwds);
            if (out == NULL) {
                Py_DECREF(func);
                Py_DECREF(module);
                return NULL;
            }
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
        }
        Py_XDECREF(func);
        Py_DECREF(module);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", name);
        return NULL;
    }

    return out;
}
