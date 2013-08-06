"""
TODO: Documentation

----
Authors: 
Eric J. Humphrey (ejhumphrey@nyu.edu)
Uri Nieto (oriol@nyu.edu)

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

import numpy as np
from sklearn.decomposition import PCA
import os
import cPickle
from scipy.cluster.vq import kmeans
import pylab as plt
#from deeplearn.core.initialization import online_vq

DC_INDEX = 450
F_DIM = 900

class Transform(object):
    
    def __init__(self, filename=''):
        object.__init__(self)
        self.filename=filename
        self.params = {}
        self.load()
        
    def __call__(self, X):
        raise NotImplementedError("Write me!")
    
    def save(self, filename=None, OVERWRITE=False):
        """
        writes the transform's param dictionary to a pickle file.
        
        filename : str, default=None
            filepath to write to, defaults to the predefined filepath (if there is one)
        """
        if filename is None:
            filename = self.filename
        if len(filename)==0:
            print """Warning! Invalid filename. Either set self.filename or provide a valid path."""
            return
        if os.path.exists(filename) and not OVERWRITE:
            print """Warning! File exists! Pass OVERWRITE=True to this method to force."""
            return
        
        fh = open(filename,'w')
        cPickle.dump(self._pack_params(),fh)
        fh.close()
        
        
    def load(self, filename=None):
        """
        load previously saved parameters. if no filename is given, attempts to load
        the default (on init) filename.
        
        """
        if filename is None:
            filename = self.filename
        
        if os.path.exists(filename):
            self._unpack_params(cPickle.load(open(filename)))
    
    def _pack_params(self):
        pass
            
    def _unpack_params(self, params):
        pass
    

class PCATransform(Transform):
    
    def __init__(self, filename=None,
                 n_components=450,
                 whiten=False,
                 l2norm=True,
                 log_coeff=5.0,
                 dc_scalar=1.0):
        
        self.n_components = n_components
        self.whiten = whiten
        self.pca = None
        self.data_shape = []
        self.log_coeff = log_coeff
        if np.abs(dc_scalar) > 0.0:
            # Produce a vector
            self.scale = np.ones([1,F_DIM])
            self.scale[0,DC_INDEX] = dc_scalar
        else:
            self.scale = np.ones(F_DIM,dtype=bool)
            self.scale[DC_INDEX] = False
        self.l2norm = l2norm
        
        if filename is None:
            filename = "PCATransform_ncE%d_whitenE%s_l2normE%s_lcE%0.1f"%(n_components,
                                                                         whiten,
                                                                         l2norm,
                                                                         log_coeff)
            filename = filename.replace(".","x") + ".pk"
        
        Transform.__init__(self, filename=filename)
        
        
    def _pack_params(self):
        return {'pca':self.pca,
                'data_shape':self.data_shape,
                'n_components':self.n_components,
                'whiten':self.whiten,
                'scale':self.scale,
                'log_coeff':self.log_coeff,
                'l2norm':self.l2norm}
    
    def _unpack_params(self, params):
        self.pca = params.get('pca')
        self.whiten = params.get('whiten')
        self.data_shape = params.get('data_shape')
        self.n_components = params.get('n_components')
        self.scale = params.get("scale")
        self.log_coeff = params.get("log_coeff")
        self.l2norm = params.get("l2norm")    
                
    def _preprocess(self,X):
        # NaN / Inf check
        X = X[np.isfinite(X.mean(axis=1))]
        if X.shape[0]==0:
            return X
        # DC Scalar
        if self.scale.squeeze()[DC_INDEX]>0:
            # If non-zero, apply
            X = X*self.scale
        else:
            # Else, select
            X = X[:,self.scale]
        
        # Apply L2-norm
        if self.l2norm:
            X = l2_norm(X)
        
        # Log-whitening
        if self.log_coeff > 0:
            X = np.log1p(self.log_coeff*X)
        return X
    
    def fit(self, X):
        self.pca = PCA(n_components=self.n_components, whiten=self.whiten)
        
        # Fit pca
        self.pca.fit(self._preprocess(X))
        
        
    def __call__(self, X):
        flatten = False
        if X.ndim==1:
            X = X[np.newaxis,...]
        
        assert X.shape[1]==F_DIM
        Z = self.pca.transform(self._preprocess(X))
        #plt.imshow(Z, interpolation="nearest", aspect="auto"); plt.show()
        if flatten:
            Z = Z.squeeze()
        return Z
    
class BasisProjection(PCATransform):
    def __init__(self, W=None, filename=None,
                 shrink=0.0,
                 act='dot',
                 act_norm=True,
                 n_components=450, 
                 whiten=False, 
                 l2norm=True, 
                 log_coeff=5.0, 
                 dc_scalar=1.0,
                 pca=None):
        """
        act : str
            one of 'dot','l2','l1'
        """
        self.act = act
        self.shrink = shrink
        self.act_norm = act_norm
        self.W = W
        
        PCATransform.__init__(self, filename=filename,
                               n_components=n_components,
                               whiten=whiten,
                               l2norm=l2norm, 
                               log_coeff=log_coeff, 
                               dc_scalar=dc_scalar)
        
        if filename is None:
            filename = "BasisProjection_kE%d_actE%s_anormE%s_shkE%0.3f"%(self.W.shape[0],
                                                                         act,
                                                                         act_norm,
                                                                         shrink)
            filename = filename.replace(".","x") + ".pk"
        self.filename = filename
        if not pca is None:
            self.pca = pca
    
    
    def _pack_params(self):
        p = PCATransform._pack_params(self)
        p.update({'act':self.act,
                  'act_norm':self.act_norm,
                  'W':self.W,
                  'shrink':self.shrink})
        return p
        
    
    def _unpack_params(self, params):
        PCATransform._unpack_params(self, params)
        self.act = params.get("act")
        self.act_norm = params.get("act_norm")
        self.W = params.get("W")
        self.shrink = params.get("shrink")
    
    def __call__(self, X):
        Z = PCATransform.__call__(self, X)
        
        if self.act == 'l2':
            A = l2(Z,self.W)
        elif self.act =='dot':
            A = np.dot(Z,self.W.T)
        else:
            raise ValueError("Unsupported activation: %s"%self.act)
        
        if self.act_norm:
            A = l2_norm(A)    
        
        if shrink>0:
            A = shrink(A,self.shrink)
            
        return A
        
class BasisProjection2(BasisProjection):
    def __init__(self, W=None, filename=None,
                 shrink=0.0,
                 act='dot',
                 act_norm=True,
                 n_components=450, 
                 whiten=False, 
                 l2norm=True, 
                 log_coeff=5.0, 
                 dc_scalar=1.0,
                 pca=None):
        """
        act : str
            one of 'dot','l2','l1'
        """
        BasisProjection.__init__(self, W,
                                 filename, 
                                 shrink, 
                                 act, 
                                 act_norm, 
                                 n_components, 
                                 whiten, 
                                 l2norm, 
                                 log_coeff, 
                                 dc_scalar,
                                 pca)
        if filename is None:
            filename = "BasisProjection2_kE%d_actE%s_shkE%0.3f_anormE%s"%(self.W.shape[0],
                                                                          act,
                                                                          shrink,
                                                                          act_norm)
            filename = filename.replace(".","x") + ".pk"
        self.filename = filename
    
    def __call__(self, X):
        Z = PCATransform.__call__(self, X)
        
        if self.act == 'l2':
            A = l2(Z,self.W)
        elif self.act =='dot':
            A = np.dot(Z,self.W.T)
        else:
            raise ValueError("Unsupported activation: %s"%self.act)
        
        if shrink>0:
            A = shrink(A,self.shrink)
        
        if self.act_norm:
            A = l2_norm(A)    
            
        return A    
        
        
        

class KMeansTransform(Transform):
    
    def __init__(self, filename='', k = 256, n_components=450, whiten=False,
                 dc_scalar=1.0, log_coeff=10.0, act_type='l2'):
        
        self.n_components = n_components
        self.whiten = whiten
        self.pca = None
        self.dict = None
        self.act_type=act_type
        self.k = k
        self.data_shape = []
        self.log_coeff = log_coeff
        if np.abs(dc_scalar) > 0.0:
            # Produce a vector
            self.scale = np.ones([1,F_DIM])
            self.scale[0,DC_INDEX] = dc_scalar
        else:
            self.scale = np.ones(F_DIM,dtype=bool)
            self.scale[DC_INDEX] = False
        
        Transform.__init__(self, filename=filename)
        
        
    def _pack_params(self):
        return {'pca':self.pca,
                'dict':self.dict,
                'k':self.k,
                'data_shape':self.data_shape,
                'n_components':self.n_components,
                'whiten':self.whiten,
                'scale':self.scale,
                'log_coeff':self.log_coeff}
    
    def _unpack_params(self, params):
        self.pca = params.get('pca')
        self.whiten = params.get('whiten')
        self.dict = params.get('dict')
        self.data_shape = params.get('data_shape')
        self.n_components = params.get('n_components')
        self.scale = params.get("scale",self.scale)
        self.log_coeff = params.get("log_coeff",self.log_coeff)
        self.k = params.get('k')    
                
        
    def fit(self, X):
        self.pca = PCA(n_components=self.n_components, whiten=self.whiten)
        
        if self.log_coeff > 0:
            X = np.log1p(self.log_coeff*X)

        if self.scale.squeeze()[DC_INDEX]>0:
            # If non-zero, apply
            X = X*self.scale
        else:
            # Else, select
            X = X[:,self.scale]
        
        self.pca.fit(X)
        self.dict = kmeans(self.pca.transform(X),self.k,iter=10)[0]
                
    def __call__(self, X):
        flatten = False
        if X.ndim==1:
            X = X[np.newaxis,...]
        
        assert X.shape[1]==F_DIM
        
        if self.log_coeff > 0:
            X = np.log1p(self.log_coeff*X)

        if self.scale.squeeze()[DC_INDEX]>0:
            Z = self.pca.transform(X*self.scale)
        else:
            Z = self.pca.transform(X[:,self.scale])
        
        if self.act_type == 'l2':
            Z = np.sqrt(np.power(Z[:,np.newaxis,:] - self.dict[np.newaxis,...],2.0).sum(axis=-1))
        else:
            raise ValueError("act_type '%s' currently unsupported"%self.act_type)
        
        if flatten:
            Z = Z.squeeze()
        
        return Z


class DrLIMTransform(Transform):
    
    def __init__(self, filename='',param_values=None):
        
        if not param_values is None:
            self.param_values = param_values.copy()
        Transform.__init__(self, filename=filename)
        
    def _pack_params(self):
        return {'param_values':self.param_values}
    
    def _unpack_params(self, params):
        self.param_values = params['param_values'].copy()        
    
    def transform(self, X):
        W0,b0 = [self.param_values[0][k] for k in ['W_fwd','b_fwd']]
        W1,b1 = [self.param_values[1][k] for k in ['W_fwd','b_fwd']]
        
        z0 = np.tanh(np.dot(X,W0) + b0[np.newaxis,:])
        return np.tanh(np.dot(z0,W1) + b1[np.newaxis,:])
        
    def __call__(self, X):
        return self.transform(X)
    
    
def load_transform(filename):
    class_string = os.path.split(filename)[-1].split('_')[0]
    return eval(class_string + """(filename='%s')"""%filename)
    
def l2_norm(x):
    flatten = False
    if x.ndim==1:
        x = x[np.newaxis,:]
        flatten = True
    u = np.sqrt(np.power(x,2.0).sum(axis=-1))
    u[u==0] = 1.0
    y = x / u[:,np.newaxis]
    if flatten:
        y = y.flatten()
    return y

def shrink(x,theta):
    """
    x : np.ndarray
    theta : knee location for shrinkage function
    """
    return np.sign(x)*((np.abs(x)-theta) + np.abs(np.abs(x)-theta))

def l2(x,y):
    flatten_x,flatten_y = False, False
    if x.ndim==1:
        x = x[np.newaxis,:]
    if y.ndim==1:
        y = y[np.newaxis,:]
        
    d = np.sqrt(np.power(x[:,np.newaxis,:]-y[np.newaxis,...],2.0).sum(axis=-1))
    if flatten_x:
        d = d[0,...]
    elif flatten_y:
        d = d[:,0]
    elif flatten_x and flatten_y:
        d = d[0]
        
    return d

def lp_distance(x,y,p):
    flatten_x,flatten_y = False, False
    if x.ndim==1:
        x = x[np.newaxis,:]
    if y.ndim==1:
        y = y[np.newaxis,:]
        
    d = np.power(np.power(np.abs(x[:,np.newaxis,:]-y[np.newaxis,...]),p).sum(axis=-1),1./p)
    if flatten_x:
        d = d[0,...]
    elif flatten_y:
        d = d[:,0]
    elif flatten_x and flatten_y:
        d = d[0]
        
    return d

"""
def fit_ovq(k, load_npy=True, iter_factor=50, k_sub=0):
    n_iter = iter_factor*k
    Z = np.load("shs_pca_subsample50k.npy")
    if load_npy:
        k = np.load("Dropbox/NYU/2013_01_Spring/research/covers/k%d.npy"%k)
    return online_vq(Z,k,eta=0.01,max_iter=n_iter,di_updates=1,init_mode='points',k_sub=k_sub)
"""