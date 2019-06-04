# DeltaSherlock. See README.md for usage. See LICENSE for MIT/X11 license info.
"""
DeltaSherlock common IO module. Useful for saving and loading fingerprint/changeset objects
"""
import pickle
import os
import tempfile
import random
import string
import time
import json
import numpy as np
from changesets import Changeset
from changesets import ChangesetRecord


class DSEncoder(json.JSONEncoder):
    """
    Provides some JSON serialization facilities for custom objects used by
    DeltaSherlock (currently supports Fingerprints, Changesets, and
    ChangesetRecords). Ex. Usage: json_str = DSEncoder().encode(my_changeset)
    """

    def default(self, o: object):
        """
        Coverts a given object into a JSON serializable object. Not to be used
        directly; instead use .encode()
        :param: o the Fingerprint or Changeset to be serialized
        :returns: a JSON serializable object to be processed by the standard Python
        JSON encoder
        """
        serializable = dict()
        # Check what kind of object o is
        if (isinstance(o, Fingerprint)):
            serializable['type'] = "Fingerprint"
            serializable['method'] = o.method.value
            serializable['labels'] = o.labels
            serializable['predicted_quantity'] = o.predicted_quantity
            serializable['array'] = o.tolist()

        elif (isinstance(o, Changeset)):
            serializable['type'] = "Changeset"
            serializable['open_time'] = o.open_time
            serializable['open'] = o.open
            serializable['close_time'] = o.close_time
            serializable['labels'] = o.labels
            serializable['predicted_quantity'] = o.predicted_quantity

            # Rescursively serialize the file change lists
            serializable['creations'] = list()
            for cs_record in o.creations:
                serializable['creations'].append(self.default(cs_record))

            serializable['modifications'] = list()
            for cs_record in o.modifications:
                serializable['modifications'].append(self.default(cs_record))

            serializable['deletions'] = list()
            for cs_record in o.deletions:
                serializable['deletions'].append(self.default(cs_record))

        elif (isinstance(o, ChangesetRecord)):
            serializable['type'] = "ChangesetRecord"
            serializable['filename'] = o.filename
            serializable['filesize'] = o.filesize
            serializable['mtime'] = o.mtime
            serializable['neighbors'] = o.neighbors

        else:
            # Unknown object. Pass it up to the parent encoder
            serializable = json.JSONEncoder.default(self, o)

        return serializable


def save_object_as_json(obj: object, save_path: str):
    """
    Basically saves a text representation of select DeltaSherlock objects to a file.
    Although less space efficient than a regular binary Pickle file, it allows for
    easier transport via network, and is MUCH less vulnerable to arbitrary code execution attacks.
    :param obj: the object to be saved (supports anything supported by DSEncoder)
    :param save_path: the full path of the file to be saved (existing files will
    be overwritten)
    """
    with open(save_path, 'w') as output_file:
        print(DSEncoder().encode(obj), file=output_file)
