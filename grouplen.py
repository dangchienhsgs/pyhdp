import pandas as pd
import tensorflow as tf

TRAIN_PATH = "data/grouplens/ua.base"
TEST_PATH = "data/grouplens/ua.test"


def read_one(path):
    f = open(path)
    _users = []
    _items = []
    _rates = []
    for user_id, item_id, rating, timestamp in _parse(f):
        if user_id not in _users:
            _users.append(user_id)

        if item_id not in _items:
            _items.append(item_id)

        _rates.append([user_id, item_id, rating, timestamp])

    return _users, _items, _rates


def _parse(data):
    for line in data:

        if not line:
            continue

        uid, iid, rating, timestamp = [int(x) for x in line.split('\t')]

        # Subtract one from ids to shift
        # to zero-based indexing
        yield uid - 1, iid - 1, rating, timestamp


if __name__ == "__main__":
    users, items, rates = read_one(TRAIN_PATH)
    print(len(users))
    print(len(items))
    print(len(rates))

    users, items, rates = read_one(TEST_PATH)
    print(len(users))
    print(len(items))
    print(len(rates))
