import numpy as np


def read_vocabs(vocab_path):
    f = open(vocab_path)
    vocabs = []
    for line in f:
        vocabs.append(line.strip())

    vocabs = np.array(vocabs)
    f.close()
    return vocabs


def read_features_items(items_path, dim):
    count = 0
    f = open(items_path)
    items = []
    for line in f:
        args = line.split(" ")

        x = np.zeros(dim, dtype=int)
        for i in range(1, len(args)):
            t = args[i].split(":")
            word = int(t[0])
            size = int(t[1])
            x[word] = size

            count += 1
        items.append(x)

    print("Total {} words ".format(count))
    return items


def read_users_like(user_path):
    f = open(user_path)

    user_likes = []
    item_max = 0
    for line in f:
        args = line.split(" ")
        user = [int(x) for x in args]
        user_likes.append(user)

        item_max = max(item_max, np.max(user))

    user_likes = np.array(user_likes)
    n_users = len(user_likes)
    n_items = item_max + 1

    return user_likes, n_items, n_users
