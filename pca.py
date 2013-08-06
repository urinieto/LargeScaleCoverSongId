"""
Because reinventing the wheel is so much fun, here is my own PCA implementation
using SVD in numpy.
Ok, I did read a lot of existing implementations, for instance:
http://stackoverflow.com/questions/1730600/principal-component-analysis-in-python
http://www.cs.stevens.edu/~mordohai/classes/cs559_s09/PCA_in_MATLAB.pdf
http://en.wikipedia.org/wiki/Singular_value_decomposition
http://en.wikipedia.org/wiki/Principal_component_analysis

My goal is to have a PCA when data is centered but not normalized, and be able
to use it with unseen data.

----
Author: 
Thierry Bertin-Mahieux (tb2332@columbia.edu)

----
License:
This code is distributed under the GNU LESSER PUBLIC LICENSE 
(LGPL, see www.gnu.org).

Copyright (c) 2012-2013 MARL@NYU.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of MARL, NYU nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""

import os
import sys
import pickle
import copy
import time
import numpy as np
from numpy.linalg import svd


class PCA(object):
    """
    implementation of a PCA, for both training and applying it
    to unseen data.
    For the moment we keep all the singular values, so we can
    apply PCA with as many dimensions as we want. If it's heavy on
    memory we'll cut.
    """

    def __init__(self, data, inline=False):
        """
        does the actual training!
        data    - M x N, M number of examples, N dimension
        inline  - if True, data is modified
        """
        # original shape
        self.original_shape = data.shape
        # save means
        self.means = data.mean(axis=0)
        # center and save the means
        if not inline:
            data = data.copy()
        self.center_newdata(data)
        data /= np.sqrt(data.shape[0]-1) # to get the variance right
        # compute SVD!
        self.U, self.d, self.Vt = svd(data, full_matrices=False)
        if self.U.shape[0] * self.U.shape[1] > 500000:
            del self.U
        # make sure the values are properly sorted
        assert np.all(self.d[:-1] >= self.d[1:])
        # get the variance
        # if we want the eigenvalues, we must run without normalizing by the number of examples
        # getting the variance makes sense when using the covariance matrix
        # instead of SVD to compute PCA
        self.variance = self.d**2
        # built time
        self.built_time = time.ctime()


    def apply_newdata(self, data, ndims=-1):
        """
        Apply PCA to new data
        By default, dimensionality is preserved
        """
        if ndims < 1:
            return np.dot(data - self.means, self.Vt.T)
        else:
            return np.dot(data - self.means, self.Vt[:ndims].T)


    def center_newdata(self, data):
        """
        center data inline (brings the column mean to zero)
        Use copy if you need to preserve the data.
        Uses the saved means, must have been computed!
        data  - M x N, M number of examples, N dimensions
        """
        data -= self.means

    
    def uncenter(self, data):
        """
        Uncenter the data inline (add backs the means)
        Use copy if you need to preserve the data.
        data  - M x N, M number of examples, N dimension
        """
        data += self.means
        

    def __repr__(self):
        """
        Quick string presentation of the data
        """
        s = 'PCA built on data shaped: %s' % str(self.original_shape)
        s += ' on %s' % self.built_time
        return s
