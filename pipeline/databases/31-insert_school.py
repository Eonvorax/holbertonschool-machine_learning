#!/usr/bin/env python3
"""
Insert a document in a collection
"""


def insert_school(mongo_collection, **kwargs):
    """
    Inserts a new document in a MongoDB collection based on kwargs.

    Args:
        mongo_collection: The pymongo collection object.
        **kwargs: Key-value pairs representing the fields and values of
            the new document.

    Returns:
        The ID of the newly created document.
    """
    return mongo_collection.insert_one(kwargs).inserted_id
