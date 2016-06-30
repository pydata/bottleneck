#include <Python.h>

static PyObject *slow_module = NULL;


PyObject *
slow(char *name, PyObject *args, PyObject *kwds)
{
    PyObject *func = NULL;
    PyObject *out = NULL;

    if (slow_module == NULL) {
        /* bottleneck.slow has not been imported during the current
         * python session. Only import it once per session to save time*/
        slow_module = PyImport_ImportModule("bottleneck.slow");
        if (slow_module == NULL) {
            PyErr_SetString(PyExc_RuntimeError,
                            "Cannot import bottleneck.slow");
            return NULL;
        }
    }

    func = PyObject_GetAttrString(slow_module, name);
    if (func == NULL) {
        PyErr_Format(PyExc_RuntimeError,
                     "Cannot import %s from bottleneck.slow", name);
        return NULL;
    }
    if (PyCallable_Check(func)) {
        out = PyObject_Call(func, args, kwds);
        if (out == NULL) {
            Py_XDECREF(func);
            return NULL;
        }
    }
    else {
        Py_XDECREF(func);
        PyErr_Format(PyExc_RuntimeError,
                     "bottleneck.slow.%s is not callable", name);
        return NULL;
    }
    Py_XDECREF(func);

    return out;
}
