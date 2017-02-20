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

from functools import partial

from into import into
import pandas as pd
import numpy as np
import gensim
import toolz as tlz
from toolz.functoolz import pipe


from superserial.utils import psql_query
from textquant.models.modelIO import (load_best_fit_model_corpus,
                                      find_best_fit_model_corpus_id)


# -----------------------------------------------------------------------------
# ## Get Data
# ### LDA Vectors


def top_model_corpus(model):
    return pipe(model,
                load_best_fit_model_corpus,
                lambda d: dict.get(d, 'corpus'))


def corpus_2_dataframe(corpus):
    corp_2_dense = partial(gensim.matutils.corpus2dense,
                           num_terms=corpus.num_terms)
    to_df = partial(into, pd.DataFrame)
    return pipe(corpus, corp_2_dense, np.rot90, to_df)


def top_model_corpus_df(model):
    return pipe(model, top_model_corpus, corpus_2_dataframe)


# -----------------------------------------------------------------------------
# ### Map Real Doc ID's to Seq


def top_model_crps_id(model):
    """
    Gets the Id of the corpus created by the top version of the model
    type supplied.
    """
    return pipe(model,
                find_best_fit_model_corpus_id,
                tlz.curried.get_in(['id']))


def top_model_seq2docs(model):
    """
    Gets the Document ID to sequence matches for the top
    version of the model type supplied.
    """
    Q = """
        SELECT doc_id AS id, seq
          FROM doc_to_sub_corpus
          WHERE sub_corpus_id = %(crps)s
        ;
        """
    return tlz.pipe('lda',
                    lambda m: psql_query(Q, {'crps': top_model_crps_id(m)}),
                    tlz.curried.groupby(lambda d: d['seq']),
                    tlz.curried.valmap(lambda d: d[0]['id']),
                    )
