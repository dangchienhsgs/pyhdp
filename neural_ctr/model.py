from keras.models import Model
from keras.layers import Dense, Input, merge
from neural_ctr.reader import *


def create_model(n_user, n_item, n_vocab, n_latent):
    user_input = Input(shape=(n_user,))
    user_latent = Dense(n_latent)(user_input)

    item_input = Input(shape=(n_item,))
    item_latent = Dense(n_latent)(item_input)

    vocab_input = Input(shape=(n_vocab,))
    vocab_latent = Dense(n_latent)(vocab_input)

    merge_user_item = merge([user_latent, item_latent], mode='concat')
    merge_user_item_2 = Dense(8)(merge_user_item)
    merge_user_item_final = Dense(1)(merge_user_item_2)

    merge_item_vocab = merge([item_latent, vocab_latent], mode='concat')
    merge_item_vocab_2 = Dense(8)(merge_item_vocab)
    merge_item_vocab_final = Dense(1)(merge_item_vocab_2)

    m = Model(input=[user_input, item_input, vocab_input],
              output=[merge_user_item_final, merge_item_vocab_final])

    m.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[0.1, 0.2], metrics=['accuracy'])

    return m


def generator(user_likes, item_info, n_user, n_item, n_vocabs):
    users = []
    items = []
    words = []

    uil = []
    iwl = []
    for i in range(0, len(users_likes)):
        for j in user_likes[i]:
            for w in range(len(item_info[j])):
                user = one_hot(i, n_user, 1)
                item = one_hot(j, n_item, 1)
                word = one_hot(w, n_vocabs, 1)

                user_item_label = 1
                item_word_label = item_info[j][w]

                users.append(user)
                items.append(item)
                words.append(word)

                uil.append(user_item_label)
                iwl.append(item_word_label)

                if len(users) == 100:
                    yield [np.array(users), np.array(items), np.array(words)], [np.array(uil), np.array(iwl)]
                    users = []
                    items = []
                    words = []

                    uil = []
                    iwl = []

        item_not_like = [x for x in range(n_item) if x not in users_likes[i]]
        item_not_like = np.random.choice(item_not_like, len(users_likes[i]))
        for j in item_not_like:
            for w in range(len(item_info[j])):
                user = one_hot(i, n_user, 1)
                item = one_hot(j, n_item, 1)
                word = one_hot(w, n_vocabs, 1)

                user_item_label = 0
                item_word_label = item_info[j][w]

                users.append(user)
                items.append(item)
                words.append(word)

                uil.append(user_item_label)
                iwl.append(item_word_label)

                if len(users) == 100:
                    yield [np.array(users), np.array(items), np.array(words)], [np.array(uil), np.array(iwl)]
                    users = []
                    items = []
                    words = []

                    uil = []
                    iwl = []


def generator_test(user_likes, item_info, n_user, n_item, n_vocabs):
    users = []
    items = []
    words = []

    uil = []
    iwl = []
    for i in range(0, len(users_likes)):
        for j in user_likes[i]:
            for w in range(len(item_info[j])):
                user = one_hot(i, n_user, 1)
                item = one_hot(j, n_item, 1)
                word = one_hot(w, n_vocabs, 1)

                user_item_label = 1
                item_word_label = item_info[j][w]

                users.append(user)
                items.append(item)
                words.append(word)

                uil.append(user_item_label)
                iwl.append(item_word_label)

                if len(users) == 100:
                    yield [np.array(users), np.array(items), np.array(words)], [np.array(uil), np.array(iwl)]
                    users = []
                    items = []
                    words = []

                    uil = []
                    iwl = []

                break


def one_hot(i, dim, value):
    x = np.zeros(dim)
    x[i] = value
    return x


if __name__ == "__main__":
    vocabs = read_vocabs("data/ctr/vocab.dat")
    item_info = read_features_items("data/ctr/mult.dat", len(vocabs))
    users_likes, num_items, num_users = read_users_like("data/ctr/cf-train-1-users.dat")
    users_likes_test, num_items, num_users = read_users_like("data/ctr/cf-test-1-users.dat")

    print("Number of users: ", num_users)
    print("Number of items: ", num_items)
    print("Number of vocabulary", len(vocabs))

    print("Total interaction train ", np.sum([len(x) for x in users_likes]))
    print("Total interaction test ", np.sum([len(x) for x in users_likes_test]))

    model = create_model(num_users, num_items, len(vocabs), 10)
    model.fit_generator(generator=generator(users_likes, item_info, num_users, num_items, len(vocabs)),
                        nb_epoch=10,
                        samples_per_epoch=340006,
                        nb_val_samples=300,
                        validation_data=generator_test(users_likes_test, item_info, num_users, num_items, len(vocabs)))

    score = model.predict_generator(
        generator=generator_test(users_likes_test, item_info, num_users, num_items, len(vocabs)),
        val_samples=1000)

    print(score)
