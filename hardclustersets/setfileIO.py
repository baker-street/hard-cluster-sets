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
import os

import toolz as tlz

from tyr.filesystem import name_to_id, list_files, new_dir
from tyr.interpreter import interpreter
from tyr.setssytem import init_new_set_file, add_to_set


CLUSTER_SETS_DIR = "kmeans"


# -----------------------------------------------------------------------------

def init_clstr_set_dir():
    new_dir(name=CLUSTER_SETS_DIR, parentid=name_to_id('/'), ownerid="ubuntu")


def get_clstr_set_dir_id():
    return name_to_id(CLUSTER_SETS_DIR)


def clstr_set_dir_id():
    try:
        return get_clstr_set_dir_id()
    except AssertionError:
        init_clstr_set_dir()
        return clstr_set_dir_id()


# -----------------------------------------------------------------------------


def create_new_clstr_set_file(name):
    init_new_set_file(name=name, settype="kmeans_based_set", parentid=clstr_set_dir_id())


def clstr_set_file_id(name):
    return name_to_id(os.path.join('/', CLUSTER_SETS_DIR, name))


def check_if_clstr_set_file_exists(name):
    try:
        clstr_set_file_id(name)
        return True
    except AssertionError:
        return False


def create_clstr_set_file_if_missing(name):
    if not check_if_clstr_set_file_exists(name):
        create_new_clstr_set_file(name)


def new_clstr_set_file(name):
    """
    creates cluster setfile and then returns the new cluster setfiles id.
    """
    create_new_clstr_set_file(name)
    return clstr_set_file_id(name)


# --------------------------------------


def load_clstr_set_doc_ids(name):
    return interpreter(os.path.join('/', CLUSTER_SETS_DIR, name))


def clstr_set_doc_ids(name):
    try:
        return load_clstr_set_doc_ids(name)
    except AssertionError:
        return set()


# --------------------------------------


def add_to_clstr_set_file(docids, setname=None, setid=None):
    """
    Add list of doc ID's to existing setfile.
    """
    if (not bool(setid)) and bool(setname):
        setid = clstr_set_file_id(setname)
    add_to_set(list(docids), setid=setid)
    return setid


# --------------------------------------


def clstr_2_setfile(clstr, docids):
    """
    assumes that 'clstr' is an 'int' and that 'docids' is a 'list'
    """
    setid = new_clstr_set_file("cluster_{}".format(clstr))
    add_to_clstr_set_file(docids=docids, setid=setid)
    return setid


def clstrs_2_setfiles(clstr2docmap):
    """
    Creates new SetFiles for each cluster.

    Accepts a dict with cluster number as the key and lists of docids as the values.
    """
    return tlz.map(lambda t: apply(clstr_2_setfile, t),
                   clstr2docmap.iteritems())


# ---------------------------


"""hardclustersets  Copyright (C) 2016  Steven Cutting"""
