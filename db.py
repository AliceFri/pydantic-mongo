import random

import pymongo
from pymongo.database import Database
from pymongo.collection import Collection

_CLIENT = None
_DB : Database = None

__all__ = [
    'register_collection',
    'get_db',
    'get_client',
    'get_collection',
]


def register_collection(client_uri, db_name):
    global _DB, _CLIENT
    _CLIENT = []
    for i in range(16):
        _CLIENT.append(pymongo.MongoClient(client_uri))
    _DB = [O[db_name] for O in _CLIENT]


def get_db() -> Database:
    if _DB is None:
        raise Exception('DB not registered')
    return random.choice(_DB)


def get_client():
    if _CLIENT is None:
        raise Exception('Client not registered')
    return random.choice(_CLIENT)


def get_collection(collection_name) -> Collection:
    return get_db()[collection_name]
