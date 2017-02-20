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


import toolz as tlz
from toolz.functoolz import pipe


from hardclustersets import utils
from hardclustersets.textquantIO import top_model_corpus_df, top_model_seq2docs
from hardclustersets.clustering import kmeans_clusters
from hardclustersets.setfileIO import clstrs_2_setfiles


# ---------------------------


@tlz.curry
def top_model_kmeans_clstrs(model, n_clusters=8, **kwargs):
    """
    Fits a kmeans model on the requested corpus (uses mini batch kmeans).
    returns the cluster labels.
    """
    return pipe(model,
                top_model_corpus_df,
                kmeans_clusters(n_clusters=n_clusters, **kwargs))


# ---------------------------


@tlz.curry
def top_model_docs2iterbl(model, iterbl):
    """
    Assigns Doc ID's to there seq's id's in the iterable.
    """
    return tlz.pipe(iterbl,
                    tlz.curried.pipe(model,
                                     top_model_seq2docs,
                                     utils.assign_seq2match))


# ---------------------------


@tlz.curry
def kmeans_top_model_corpus(model, **kwargs):
    """
    Creates the clusters using KMeans and returns the Doc ID's
    assigned to each new cluster.
    """
    return tlz.pipe(model,
                    top_model_kmeans_clstrs(**kwargs),
                    top_model_docs2iterbl(model),
                    utils.group_by_item,
                    )


# ---------------------------


def new_clstr_setfiles(model, **kwargs):
    """
    Creates the clusters using KMeans then creates new SetFiles for each cluster
    using the doc ID's.
    """
    return tlz.pipe(model,
                    kmeans_top_model_corpus(**kwargs),
                    clstrs_2_setfiles)


# ---------------------------


"""hardclustersets  Copyright (C) 2016  Steven Cutting"""
