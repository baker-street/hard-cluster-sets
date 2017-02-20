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


import itertools

import toolz as tlz


def assign_seq(iterbl):
    """
    Assign sequence id's (0,1,2,n) to the supplied iterable.
    Returns a stream of tuples.
    """
    return tlz.map(tuple, itertools.izip(itertools.count(), iterbl))


@tlz.curry
def assign_seq2match(seq2match, iterbl):
    return tlz.pipe(iterbl,
                    assign_seq,
                    tlz.curried.map(lambda t: (seq2match[t[0]], t[1])),
                    )


def group_by_item(listotpls, x=1, y=0):
    """
    When given a list of 2D tuples, it performs a groupby on the item
    at index 'x' creating a dict with 'x' as the key and the items at index 'y'
    placed in a list as the value.
    """
    return tlz.pipe(listotpls,
                    tlz.curried.groupby(lambda t: t[x]),
                    tlz.curried.valmap(tlz.compose(list,
                                                   tlz.curried.map(lambda t: t[y]))),
                    )
