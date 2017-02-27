import pickle

import numpy as np
import pandas as pd
from clean import clean_presets
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Lib to lemmatize
lemmatizer = WordNetLemmatizer()

'''
polarity 0 = negative. 2 = neutral. 4 = positive.
id
date
query
user
tweet
'''

# Clean all local files
clean_presets()


def decrease_train_data(fin, fout):
    outfile = open(fout, 'a')
    counter = 0
    lines = 0
    with open(fin, buffering=200000, encoding='latin-1') as f:
        try:
            for line in f:

                # Save every 10 element
                if (counter % 10) == 0:
                    outfile.write(line)
                    lines += 1
                    print(lines)
                counter += 1
        except Exception as e:
            print(str(e))
    outfile.close()


decrease_train_data('data/train-data.csv', 'data/train-data-reduced.csv')


def init_process(fin, fout):
    print('init_process')
    outfile = open(fout, 'a')
    with open(fin, buffering=200000, encoding='latin-1') as f:
        try:
            for line in f:

                # Replace all parenthesis with empty spaces
                line = line.replace('"', '')

                # Get the polarity by splitting the value
                initial_polarity = line.split(',')[0]

                # 4 means positive [0,1]
                # 0 means negative [1,0]
                if initial_polarity == '0':
                    initial_polarity = [1, 0]
                elif initial_polarity == '4':
                    initial_polarity = [0, 1]

                # Neutral which is 2 gets ignored
                if initial_polarity != '2':
                    # Get the tweet from line
                    tweet = line.split(',')[-1]

                    # Combine the polarity and the tweet together
                    outline = str(initial_polarity) + ':::' + tweet

                    # Write line to file
                    outfile.write(outline)
        except Exception as e:
            print(str(e))
    outfile.close()


init_process('data/test-data.csv', 'data/test_set.csv')
init_process('data/train-data-reduced.csv', 'data/train_set.csv')


def create_lexicon(fin):
    print('create_lexicon')
    lexicon = []
    with open(fin, 'r', buffering=100000, encoding='latin-1') as f:
        try:
            counter = 1
            for line in f:
                counter += 1

                # Read only every 2500 line from file
                if (counter / 250.0).is_integer():
                    # Get the tweet
                    tweet = line.split(':::')[1]

                    # Tokenize the tweet
                    words = word_tokenize(tweet)

                    # Lemmatize the words
                    words = [lemmatizer.lemmatize(i) for i in words]

                    # Append new words in lexicon
                    lexicon = list(set(lexicon + words))

                    print(counter, len(lexicon))
        except Exception as e:
            print(str(e))

    with open('data/lexicon.pickle', 'wb') as f:
        pickle.dump(lexicon, f)


create_lexicon('data/train_set.csv')


def convert_to_vec(fin, fout, lexicon_pickle):
    print('convert_to_vec')
    with open(lexicon_pickle, 'rb') as f:
        lexicon = pickle.load(f)
    outfile = open(fout, 'a')
    with open(fin, buffering=20000, encoding='latin-1') as f:
        counter = 0
        for line in f:
            counter += 1

            # Grap label and tweet
            label = line.split(':::')[0]
            tweet = line.split(':::')[1]

            # Tokenize Tweet
            current_words = word_tokenize(tweet.lower())

            # Lemmatize Words
            current_words = [lemmatizer.lemmatize(i) for i in current_words]

            # Create a feature set according to size of lexicon
            features = np.zeros(len(lexicon))

            # Set feature values according to tweet words
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())

                    # Increment word index
                    features[index_value] += 1

            # Generate a feature list
            features = list(features)

            # Write the list and label to file
            outline = str(features) + '::' + str(label) + '\n'
            outfile.write(outline)

        print(counter)


convert_to_vec('data/test_set.csv', 'data/test_set_vec.csv', 'data/lexicon.pickle')


def shuffle_data(fin):
    print('shuffle_data')
    df = pd.read_csv(fin, error_bad_lines=False)
    df = df.iloc[np.random.permutation(len(df))]
    print(df.head())
    df.to_csv('data/train_set_shuffled.csv', index=False)


shuffle_data('data/train_set.csv')
