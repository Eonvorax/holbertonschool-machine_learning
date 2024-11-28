#!/usr/bin/env python3
"""
List all documents in a collection
"""


def list_all(mongo_collection):
    """
    Lists all documents in a MongoDB collection.

    Args:
        mongo_collection: The pymongo collection object.

    Returns:
        A list of all documents in the collection, or an empty
        list if none exist.
    """
    return list(mongo_collection.find())
