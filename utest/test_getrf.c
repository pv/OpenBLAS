/*****************************************************************************
Copyright (c) 2011-2016, The OpenBLAS Project
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
   3. Neither the name of the OpenBLAS project nor the names of 
      its contributors may be used to endorse or promote products 
      derived from this software without specific prior written 
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

**********************************************************************************/

#include "openblas_utest.h"


void BLASFUNC(dgetri)(blasint*, double*, blasint*, blasint*, double*, blasint*, blasint*);


/* https://github.com/xianyi/OpenBLAS/issues/1533 */
CTEST(getrf, bug_1533){
  double A[100*100];
  blasint m = 100, n = 100, lda = 100, lwork = 100, info = 0;
  blasint ipiv[100];
  double *work, work_tmp;

  double alpha = 1 - 1e-6;
  double tol = 1e-9;
  int i, j;

  /* Matrix with one upper diagonal */
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      if (i == j) {
        A[i + m*j] = 1;
      } else if (i == j - 1) {
        A[i + m*j] = alpha;
      }
      else {
        A[i + m*j] = 0;
      }
    }
  }
  
  BLASFUNC(dgetrf)(&m, &n, A, &lda, ipiv, &info);
  if (info != 0) CTEST_ERR("%s:%d info != 0", __FILE__, __LINE__);

  lwork = -1;
  BLASFUNC(dgetri)(&m, A, &lda, ipiv, &work_tmp, &lwork, &info);
  if (info != 0) CTEST_ERR("%s:%d info != 0", __FILE__, __LINE__);

  lwork = (int)(1.01 * work_tmp);
  if (lwork < 0) CTEST_ERR("%s:%d bad lwork %d", __FILE__, __LINE__, lwork);
  work = malloc(sizeof(double) * lwork);
  if (work == NULL) CTEST_ERR("%s:%d malloc", __FILE__, __LINE__);
  BLASFUNC(dgetri)(&m, A, &lda, ipiv, work, &lwork, &info);
  if (info != 0) CTEST_ERR("%s:%d info != 0", __FILE__, __LINE__);
  free(work);

  /* The inverse is known: check it */
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      double expected;

      if (j >= i) {
        expected = pow(-alpha, j - i);
      }
      else {
        expected = 0;
      }

      if (!(abs(A[i + m*j] - expected) <= tol)) {
        CTEST_ERR("A[%d,%d] = %g != %g (expected)", i, j, A[i + m*j], expected);
      }
    }
  }
}
