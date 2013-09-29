Large Scale Cover Song Identification
=====================================

Source code for the article "Data Driven and Discriminative Projections for 
Large-Scale Cover Song Identification" by Eric J. Humphrey, Oriol Nieto, and 
Juan P. Bello. ISMIR, Curitiba, Brazil, 2013.

This project includes three main tasks:

- *Binary Task*: Analyze 500 tracks as described in (Thierry et al. 2012).
- *Cover Song ID in Second Hand Song Training Set* (~12,000 songs on themselves).
- *Cover Song ID in Second Hand Song Test Set* (~5,000 songs on 1,000,000).

Running the Tasks
=================

Binary Task
-----------

This task compares one given track against two others and decides which one of
the two is actually a cover of the given track. The list of tracks for this task 
is found in the file `SHS/list_500queries.txt`.

To run the task without any dimensionality reduction (i.e. each track is 
represented by the full 12x75 vector), type:

	./binary_task.py path_to_MSD

The result, as reported in (Thierry et al. 2012), should be *82.0%*.

This package already provides a previously learned Principal Component Analysis
transformation (`models/pca_250Kexamples_900dim_nocovers.pkl`). To run the task 
with a PCA of 50 components:

	./binary_task.py path_to_MSD -pca models/pca_250Kexamples_900dim_nocovers.pkl 50

This should result in *82.2%*, as reported in (Thierry et al. 2012).

To run the test using a dictionary to project the data into a (hopefully) more
separable space, we include a dictionary 
(`models/BasisProjection2_kE2045_actEdot_shkE0x200_anormETrue.pk`) that can be
used like this:

	./binary_task.py path_to_MSD -dictfile models/BasisProjection2_kE2045_actEdot_shkE0x200_anormETrue.pk

The result should be *81.2%*.

To add Linear Discriminative Analysis to the new projected space:

	./binary_task.py path_to_MSD -dictfile models/BasisProjection2_kE2045_actEdot_shkE0x200_anormETrue.pk -lda models/lda-kE2045-shkE0x200.pk n

where `n` is the index number of the model. In the given model (`models/lda-kE2045-shkE0x200.pk`), n = 0 represents 50 dimensions, n = 1 is 100 dimensions, and
n = 2 is 200 dimensions. As an example, with 200 dimensions (n = 2) the result
should be *94.4%*.

Cover Song ID in Training
-------------------------

This task computes the Mean Average Precision (MAP) and the Average Rank (AR) of 12,960 tracks from the Second Hand Song dataset. These tracks are the ones selected
for _training_, and are listed in `SHS/shs_dataset_train.txt`.

To run the task without any dimensionality reduction (i.e. each track is 
represented by the full 12x75 vector), type:

	./cover_id_train.py path_to_MSD

The result should be exactly the same as the one reported by (Thierry et al.
2012): AR = 3096.7, MAP = 8.91%.

To apply PCA of 200 on these full features:

	./cover_id_train.py path_to_MSD -pca models/pca_250Kexamples_900dim_nocovers.pkl 200

which should result in AR = 3005.1, MAP = 9.475%, as reported in (Thierry et 
al. 2012).

To apply a dictionary of k = 2045:
	
	./cover_id_train.py path_to_MSD -dictfile models/BasisProjection2_kE2045_actEdot_shkE0x200_anormETrue.pk

This should result in AR = 3026, MAP = 5.51%, as reported in (Humphrey et al.
2013).

To apply LDA of 200 dimensions:

	./cover_id_train.py path_to_MSD -dictfile models/BasisProjection2_kE2045_actEdot_shkE0x200_anormETrue.pk -lda 
	models/lda-kE2045-shkE0x200.pk 2

Which should result in a similar result than AR = 1880, MAP = 28.33%, as reported 
in (Humphrey et al. 2013).


Cover Song ID in Test
---------------------

This task compares the 5236 tracks from the test SHS dataset against all the
1,000,000 tracks from the MSD. Since computing the features for the entire MSD
takes a long time, the code is written to be run in N different threads, so that 
the work can be split in N different processors.

To compute all the 1,000,000 codes with 10 processors, type:

	./cover_id_test.py path_to_MSD -dictfile models/BasisProjection2_kE2045_actEdot_shkE0x200_anormETrue.pk -lda models/lda-kE2045-shkE0x200.pk -outdir msd_feats -N 10

Once all the features are computed inside a dir (e.g. `msd_feats`), we can
compute the score by:

	./cover_id_test.py path_to_MSD -codes msd_feats/ n

where `n` is the index number of the LDA model. In the given model (`models/lda-kE2045-shkE0x200.pk`), 
n = 0 represents 50 dimensions, n = 1 is 100 dimensions, and
n = 2 is 200 dimensions. For this script to optimally run, we will need as 
much RAM memory as possible. The script works fine with 20GB of RAM. We also
tried to run it with 6GB, but it took around 2 hours to load all the features,
instead of a couple of minutes.

Note that in this task we can't compute the entire full
features code (12x75, or, if projected to a higher dimensional space, 2045), 
since it would take too long to run.





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

Multi-core processor. At least 10 units are recommended.

And as much RAM as possible. 16GB is recommended.


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