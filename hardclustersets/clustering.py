# -*- coding: utf-8 -*-
__title__ = 'hardclustersets'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@linux.com'
__created_on__ = '01/30/2016'
__copyright__ = "hardclustersets  Copyright (C) 2016  Steven Cutting"
__credits__ = ["Steven Cutting"]
__license__ = "GPL3"
__maintainer__ = "Steven Cutting"
__email__ = 'steven.e.cutting@linux.com'

import copy

from sklearn.cluster import MiniBatchKMeans
import toolz as tlz
from toolz.functoolz import pipe


@tlz.curry
def kmeans_clusters(corpusdf, n_clusters=8, random_state=1, n_init=100, **kwargs):
    """
    Fits a kmeans model on the supplied corpus (uses mini batch kmeans).
    returns the cluster labels.
    """
    mkmeans_m = MiniBatchKMeans(n_clusters=n_clusters,
                                random_state=random_state,
                                n_init=n_init)
    return pipe(mkmeans_m,
                lambda mdl: mdl.fit(corpusdf),
                lambda mdl: mdl.labels_)


@tlz.curry
def kmeans_clstrs_with_corpus(corpusdf, **kwargs):
    """
    Not needed.

    Returns corpus dataframe with kmeans clusters added as a column.
    """
    return pipe(corpusdf,
                copy.deepcopy,
                lambda crps: (crps, kmeans_clusters(crps, **kwargs)),
                lambda args: args[0].assign(clusters=args[1])
                )
