#include "numpy/arrayobject.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <rate_solvers.h>


#define pixrad 0.0002181661564992912    /*angular pixel size in ardians*/
#define DL 0.595
#define F 2693.


int unpack_pix_index(int i, int j)
{
    return ((abs(i) > abs(j)) ? (int) ((i*i + abs(i))/2 + abs(j)) : (int) ((j*j + abs(j))/2 + abs(i))); 
}

void vec_to_inpix_coord(double * vec, double x0, double y0, long *rawx, long *rawy, double * x, double * y, int * k)
{
        int i, j; 
        i = (int)(*rawx - x0 + ((*rawx > x0) ? 0.5 : -0.5));
        j = (int)(*rawy - y0 + ((*rawy > y0) ? 0.5 : -0.5));
        if (abs(i) < abs(j)) 
        {
                *x = (i > 0) ? (vec[1]/vec[0] - (*rawx - 23.5)/F) : ((*rawx -23.5)/F - vec[1]/vec[0]);
                *y = (j > 0) ? (-vec[2]/vec[0] - (*rawy - 23.5)/F) : (vec[2]/vec[0] + (*rawy - 23.5)/F);
        }else{
                *y = (i > 0) ? (vec[1]/vec[0] - (*rawx - 23.5)/F) : ((*rawx -23.5)/F - vec[1]/vec[0]);
                *x = (j > 0) ? (-vec[2]/vec[0] - (*rawy - 23.5)/F) : (vec[2]/vec[0] + (*rawy - 23.5)/F);
        };
        *k = unpack_pix_index(i, j);
}


void inpix_vec_to_inpix_coord(double * vec, long *rawx, long *rawy, double * x, double * y)
{
        /*
         the iPSF is stored only for 1/8 of detectors pixels -- upper left triangle of uppepr right  quantile
         this is done assuming iPSF symmetry (which is not exact since optical axis does not located at the  center of pixel for each specific detector
         nevertheless we use this simplification
         that means that by swaping and changing signs of the coordinates centered at pixel we can treat any pixel of the detector
         the rule of such treatment is described bellow:
         rawx, rawy -- pixel coordinates relative to the pixel centered at the optical axis
         vec -- vec to estimate iPSF value in the pixel original coordinate system 
         */
        if (abs(*rawy) > abs(*rawx)) 
        {
                *y = (*rawx > 0) ? vec[1]/vec[0]*F  : -vec[1]/vec[0]*F;
                *x = (*rawy > 0) ? -vec[2]/vec[0]*F : vec[2]/vec[0]*F;
                //printf("check case1 %d %d %f %f \n", *rawx, *rawy, *x, *y);
        }else{
                *x = (*rawx > 0) ? vec[1]/vec[0]*F  : -vec[1]/vec[0]*F;
                *y = (*rawy > 0) ? -vec[2]/vec[0]*F : vec[2]/vec[0]*F;
                //printf("check case2 %d %d %f %f \n", *rawx, *rawy, *x, *y);
        };
}

static PyObject * get_pix_coord_for_urdn(PyObject *self, PyObject *args)
{
        PyArrayObject *rawx, *rawy, *vec;
        double x0, y0;

        if (!PyArg_ParseTuple(args, "OOOdd", &rawx, &rawy, &vec, &x0, &y0)) return NULL;
        printf("input read %d %d %f %f\n", *((long*)rawx->data), *((long*)rawy->data), x0, y0); 

        printf("dims %d\n" , rawx->dimensions[0]);
        PyArrayObject * x = PyArray_SimpleNew(1, rawx->dimensions, NPY_DOUBLE);
        PyArrayObject * y = PyArray_SimpleNew(1, rawx->dimensions, NPY_DOUBLE);
        PyArrayObject * k = PyArray_SimpleNew(1, rawx->dimensions, NPY_INT);
        int ctr;
        for (ctr=0; ctr < rawx->dimensions[0]; ctr++)
        {
                vec_to_inpix_coord((double*)vec->data + ctr*3, x0, y0, (int*)rawx->data + ctr, (int*)rawx->data + ctr, (double*)x->data + ctr, (double*)y->data + ctr, (int*)k->data + ctr); 

        };
        PyObject *res = Py_BuildValue("OOO", k, x, y);
        return res;
}

static PyObject * get_unipix_fast_index(PyObject *self, PyObject *args)
{
        PyArrayObject *i, *j, *vec;
        double xs, ys, dx, dy;
        int xsize, ysize, tile; 

        if (!PyArg_ParseTuple(args, "OOOdidi", &i, &j, &vec, &dx, &xsize, &dy, &ysize)) return NULL;
        xs = -dx*xsize/2.;
        ys = -dy*ysize/2.;

        tile = (vec->dimensions[0] > i->dimensions[0]) ? ((int) vec->dimensions[0]/i->dimensions[0]) : vec->dimensions[0];

        //printf("dims %d %f %f %d %f %f %d\n" , i->dimensions[0], dx/2.*((double)xsize + 1.), dx, xsize, xs, ys, ysize);
        npy_intp snew = {vec->dimensions[0]};
        PyArrayObject * mask = PyArray_SimpleNew(1, &snew, NPY_BOOL);
        //PyArrayObject * mask = PyArray_SimpleNew(1, {vec->dimensions[0]}, NPY_BOOL);
        long ctr, msum=0; 
        double x, y; 
        bool * maskd = (bool*)mask->data;
        int * idx1d = (int*)malloc(sizeof(int)*vec->dimensions[0]);
        int * idx2d = (int*)malloc(sizeof(int)*vec->dimensions[0]);
        long * iptr = (long*) i->data;
        long * jptr = (long*) i->data;
        double * vdata = (double*) vec->data;
        Py_BEGIN_ALLOW_THREADS;

        for (ctr=0; ctr < vec->dimensions[0]; ctr++)
        {
                inpix_vec_to_inpix_coord(vdata + ctr*3, iptr + (int) (ctr%tile), jptr + (int) (ctr%tile), &x, &y);
                if ((x > xs) && (x < -xs))
                {
                        if ((y > ys) && (y < -ys))
                        {
                                *(idx1d + msum) = (int)((x - xs)/dx);
                                *(idx2d + msum) = (int)((y - ys)/dy);
                                *(maskd + ctr) = true;
                                msum += 1;
                                //printf("%d %f %f %d %d %f %f %f %f\n", ctr, x, y, *(idx1d + msum - 1), *(idx2d + msum - 1), xs, x, (x - xs)/dx, (y - ys)/dy);
                        }else{
                                *(maskd + ctr) = false;
                        };
                }else{
                        *(maskd + ctr) = false;
                };
                //printf("%d %f %f %d %d %d %d %f %f\n", ctr, x, y, *(idx1d + msum - 1), *(idx2d + msum - 1), *((long*)i->data + ctr), *((long*)j->data + ctr), *((double*)vec->data + ctr*3), *((double*)vec->data + ctr*3 + 1));

        };
        Py_END_ALLOW_THREADS;

        npy_intp sm = {msum};
        PyArrayObject * idx1 = PyArray_SimpleNew(1, &sm, NPY_INT);
        PyArrayObject * idx2 = PyArray_SimpleNew(1, &sm, NPY_INT);
        for (ctr=0; ctr < msum; ctr++)
        {
                *((int*)idx1->data + ctr) = *(idx1d + ctr);
                *((int*)idx2->data + ctr) = *(idx2d + ctr);
        }
        PyObject *res = Py_BuildValue("OOOi", mask, idx1, idx2, msum);
        return res;
}

void rmul(double * mat, double *vec, double *out)
{
        out[0] = mat[0]*vec[0] + mat[3]*vec[1] + mat[6]*vec[2];
        out[1] = mat[1]*vec[0] + mat[4]*vec[1] + mat[7]*vec[2];
        out[2] = mat[2]*vec[0] + mat[5]*vec[1] + mat[8]*vec[2];
};

double * psfvalfromptr(double * psfdata, npy_intp * dims, int eidx, int k, int xi, int yi)
{
        //printf("check didx %d %.2e\n", ((dims[1]*eidx + k)*dims[2] + xi)*dims[3] + yi, *( psfdata + ((dims[1]*eidx + k)*dims[2] + xi)*dims[3] + yi));
        return psfdata + ((dims[1]*eidx + k)*dims[2] + xi)*dims[3] + yi;
};

static PyObject * solve_for_locations(PyObject *self, PyObject *args)
{
        PyArrayObject *i, *j, *eidx, *vec, *rmat, *smat, *pk, *emap;
        double xs, ys, dx, dy;
        int xsize, ysize, loc, k; 

        if (!PyArg_ParseTuple(args, "OOOOOOOOdidi", &i, &j, &eidx, &rmat, &pk, &vec, &emap, &smat, &dx, &xsize, &dy, &ysize)) return NULL;
        xs = -dx*xsize/2.;
        ys = -dy*ysize/2.;

        //printf("dims %d %f %f %d %f %f %d\n" , i->dimensions[0], dx/2.*((double)xsize + 1.), dx, xsize, xs, ys, ysize);
        npy_intp snew = {vec->dimensions[0]};
        PyArrayObject * cmap = PyArray_SimpleNew(1, &snew, NPY_DOUBLE);
        PyArrayObject * pmap = PyArray_SimpleNew(1, &snew, NPY_DOUBLE);

        double * lvec = (double*)malloc(sizeof(double)*3);
        double * bw = (double*)malloc(sizeof(double)*i->dimensions[0]);

        Py_BEGIN_ALLOW_THREADS;

        double * cmapd = (double*) cmap->data;
        double * pmapd = (double*) pmap->data;
        double * smatd = (double*) smat->data;
        double * pkd = (double*) pk->data;
        long * iptr = (long*)i->data;
        long * jptr = (long*)j->data;


        long ctr, msum=0; 
        double x, y, lkl; 
        double * invec;
        long * eidxd = (long*) eidx->data;

        double pval, eloc, r0; 
        int idx1d, idx2d; 


        for (loc=0; loc < vec->dimensions[0]; loc++)
        {
                msum = 0;
                r0 = 1.;
                invec = (double*) vec->data + loc*3;
                //printf("init vec %f %f %f\n", invec[0], invec[1], invec[2]);
                for (ctr=0; ctr < i->dimensions[0]; ctr++)
                {
                        rmul((double*) rmat->data + ctr*9, invec, lvec);
                        inpix_vec_to_inpix_coord(lvec, iptr + ctr, jptr + ctr, &x, &y);
                        //printf("final vec %f %f\n", lvec[1], lvec[2]);
                        if ((x > xs) && (x < -xs))
                        {
                                if ((y > ys) && (y < -ys))
                                {
                                        k = unpack_pix_index( * (iptr + ctr),  * (jptr + ctr));
                                        idx1d = (int)((x - xs)/dx);
                                        idx2d = (int)((y - ys)/dy);
                                        pval = * psfvalfromptr(smatd, smat->dimensions, k, * (eidxd + ctr), idx1d, idx2d); 
                                        pval = pval * (*(pkd + ctr));
                                        if (pval > 1e-10)
                                        {
                                                bw[msum] = pval;
                                                //printf("%d %f %f %f\n", ctr, x, y, pval);
                                                msum += 1;
                                        };
 
                                };
                        };

                };
                if (msum > 0)
                {
                        eloc = (double) *((double*) emap->data + loc);
                        pval = get_phc_solution_pkr((double) msum, eloc, bw, msum);
                        *(cmapd + loc) = pval*eloc; 
                        lkl = 1.;
                        for (ctr=0; ctr < msum; ctr ++)
                        {
                                lkl = lkl*(pval*bw[ctr] + 1.);
                        }
                        //printf("lkl %f %f\n", lkl, log(lkl));
                        *(pmapd + loc) = log(lkl); //get_lkl_pkr(pval, bw, msum);
                        //printf("photons %d %d %f %f %f %f\n", loc, msum, pval, eloc, get_lkl_pkr(pval, bw, msum), *(pmapd + loc));
                }else{
                        *(cmapd + loc) = 0.;
                        *(pmapd + loc) = 0.;
                };
        };

        Py_END_ALLOW_THREADS;

        free(bw);
        free(lvec);

        PyObject *res = Py_BuildValue("OO", cmap, pmap);
        return res;
}



static PyMethodDef PSFMethods[] = {
        {"get_pix_coord_for_urdn", get_pix_coord_for_urdn, METH_VARARGS, "get coordinates centered at provided pixel"}, 
        {"get_unipix_fast_index", get_unipix_fast_index, METH_VARARGS, "get coordinates within pixel based on its coordinates"}, 
        {"solve_for_locations", solve_for_locations, METH_VARARGS, "get coordinates within pixel based on its coordinates"}, 
        {NULL, NULL, 0, NULL}
}; 

static struct PyModuleDef psf_c_module = {
        PyModuleDef_HEAD_INIT, 
        "psf_functions", 
        NULL, 
        -1, 
        PSFMethods
};

PyMODINIT_FUNC PyInit_psf_functions(void)
{
        assert(! PyErr_Occurred());
        if (PyErr_Occurred()) {return NULL;}
        import_array();
        return PyModule_Create(&psf_c_module);
};
