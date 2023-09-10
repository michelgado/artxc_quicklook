#include "numpy/arrayobject.h"
#include <stdlib.h>
#include <stdio.h>
#include <rate_solvers.h>



void unravel_array(fptr solvfun, double * s, double *b, double *e, double * r, long * sizes, int size, double * res)
{
        int i = 0; 
        double *sl = s, *bl = b;
        for (i = 0; i < size; i++)
        {
                //printf("starting %d ", i);
                res[i] = (*solvfun)(r[i], e[i], sl, bl, sizes[i]);
                //printf("%d %d %e\n", i, sizes[i], res[i]);
                sl = sl + sizes[i];
                bl = bl + sizes[i];
        }
        //printf("ok loop is finished?\n");
        return 1;
}



static PyObject * photon_solver(PyObject *self, PyObject *args)
{
        PyArrayObject *s, *b, *r, *e, *sizes;

        if (!PyArg_ParseTuple(args, "OOOOO", &s, &b, &r, &e, &sizes)) return NULL;
        Py_BEGIN_ALLOW_THREADS;
        unravel_array(&get_phc_solution, (double*)s->data, (double*)b->data, (double*)e->data, (double*)r->data, (long*)sizes->data, r->dimensions[0], (double*)r->data);
        Py_END_ALLOW_THREADS;
        Py_RETURN_NONE;
}


static PyObject * brent_solver(PyObject *self, PyObject *args)
{
        PyArrayObject *s, *b, *r, *e, *sizes;

        if (!PyArg_ParseTuple(args, "OOOOO", &s, &b, &r, &e, &sizes)) return NULL;
        Py_BEGIN_ALLOW_THREADS;
        unravel_array(&brent, (double*)s->data, (double*)b->data, (double*)e->data, (double*)r->data, (long*)sizes->data, r->dimensions[0], (double*)r->data);
        Py_END_ALLOW_THREADS;
        Py_RETURN_NONE;
}



static PyMethodDef SolverMethods[] = {
        {"get_phc_solution", photon_solver, METH_VARARGS, "get source rate solution with photon convolution procejure"}, 
        {"get_brent_solution", brent_solver, METH_VARARGS, "get source rate solution with classical brent solver"}, 
        {NULL, NULL, 0, NULL}
}; 

static struct PyModuleDef solvermodule = {
        PyModuleDef_HEAD_INIT, 
        "src_rate_solvers", 
        NULL, 
        -1, 
        SolverMethods
};

PyMODINIT_FUNC PyInit_src_rate_solvers(void)
{
        assert(! PyErr_Occurred());
        if (PyErr_Occurred()) {return NULL;}
        return PyModule_Create(&solvermodule);
};
