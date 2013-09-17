Large Scale Cover Song Identification
=====================================

Source code for the article "Data Driven and Discriminative Projections for 
Large-Scale Cover Song Identification" by Eric J. Humphrey, Oriol Nieto, and 
Juan P. Bello. ISMIR, Curitiba, Brazil, 2013.

This project includes three main tasks:

- Binary Task: Analyze 500 tracks as described in (Thierry et. al 2012).
- Cover Song ID in Second Hand Song Training Set (~12,000 songs on themselves).
- Cover Song ID in Second Hand Song Test Set (~5,000 songs on 1,000,000).

Features
========

Binary Task
-----------

This task compares one given track against two others and decides which one of
the two is actually a cover of the given track. The list of tracks for this task 
is found in the file `SHS/list_500queries.txt`.

To run the task without any dimensionality reduction (i.e. each track is 
represented by the full 12x75 vector), type:

	./binary_task.py path_to_MSD

If you add the flag `-orig`, the result, as reported in (Thierry et. al 2012), should be 82.0%.

This package already provides a previously learned Principal Component Analysis
transformation (`models/pca_250Kexamples_900dim_nocovers.pkl`). To run the task 
with a PCA of 50 components:

	./binary_task.py path_to_MSD -pca models/pca_250Kexamples_900dim_nocovers.pkl 50

This should result in 82.2% if using the `-orig` flag, as reported in (Thierry 
et. al 2012).

To run the test using a dictionary to project the data into a (hopefully) more
separable space:

	./binary_task.py MSD -dictfile models/BasisProjection2_kE2045_actEdot_shkE0x200_anormETrue.pk

To add Linear Discriminative Analysis to the new projected space:

	./binary_task.py MSD -dictfile models/BasisProjection2_kE2045_actEdot_shkE0x200_anormETrue.pk -lda models/lda-kE2045-shkE0x200.pk n

where `n` is the index number of the model. In the given model (`models/lda-kE2045-shkE0x200.pk`), n == 0 represents 50 dimensions, n == 1 is 100 dimensions, and
n == 2 is 200 dimensions.


Requirements
============

Million Song Dataset:
http://labrosa.ee.columbia.edu/millionsong/

Numpy:
http://www.numpy.org/

Scipy:
http://www.scipy.org/

PyLab:
http://wiki.scipy.org/PyLab

Scikit-Learn:
http://scikit-learn.org/stable/


References
==========

Humphrey, E. J., Nieto, O., & Bello, J. P. (2013). Data Driven and 
Discriminative Projections for Large-Scale Cover Song Identification. 
In Proc. of the 14th International Society for Music Information Retrieval 
Conference. Curitiba, Brazil.

Bertin-Mahieux, T., & Ellis, D. P. W. (2012). Large-Scale Cover Song 
Recognition Using The 2D Fourier Transform Magnitude. In Proc. of the 13th 
International Society for Music Information Retrieval Conference (pp. 241-246).
Porto, Portugal.

Acknowledgments
===============

Thanks to Thierry Bertin-Mahieux (tb2332@columbia.edu) for sharing his code
and letting us publish it under the LGPL.

License
=======

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