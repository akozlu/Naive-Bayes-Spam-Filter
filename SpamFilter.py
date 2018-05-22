
import email
from itertools import *
from collections import *
import math
import unittest
import os


def load_tokens(email_path):

    fileInstance = open(email_path, 'r')
    messageObject = email.message_from_file(fileInstance)
    EmailIterator = email.iterators.body_line_iterator(messageObject)
    fileInstance.close()
    for line in EmailIterator:

        result = [words.lower()
                  for line in EmailIterator for words in line.split()]

    return result


def log_probs(email_paths, smoothing):

    # get the token list
    tokens = [result for path in email_paths for result in load_tokens(path)]

    # Total number of words
    NumberOfWords = len(tokens)
    # Determine the size of our vocab set
    word_counter = Counter(tokens)

    V = len(word_counter)

    # Formula based on lecture slides
    denominator = (NumberOfWords + (smoothing * (V + 1)))

    # create the log dictionary based on the formula

    log_dictionary = {word: math.log((word_counter[word] + smoothing) / denominator) for word in word_counter}

    # special case for unknowns

    log_dictionary["<UNK>"] = math.log(smoothing / denominator)

    return log_dictionary


class SpamFilter(object):

    def __init__(self, spam_dir, ham_dir, smoothing):

        # get list of spam & ham files
        spam_files = os.listdir(spam_dir)
        ham_files = os.listdir(ham_dir)

        # create the ham & spam paths

        spam = [spam_dir + "/" + s for s in spam_files]
        ham = [ham_dir + "/" + s for s in ham_files]

        # size of each directory to compute class probabilities

        spamSize = len(spam_files)
        hamSize = len(ham_files)

        # create probability dictionaries

        self.spam_dictionary = log_probs(spam, smoothing)
        self.ham_dictionary = log_probs(ham, smoothing)

        # compute class probability of ham and spam
        self.prob_spam = math.log(
            spamSize / float(spamSize + hamSize))

        self.prob_ham = math.log(1 - self.prob_spam)

    def is_spam(self, email_path):

        # Uses the approach offered in lecture slide #31 of Naive Bayes

        word_counter = Counter(load_tokens(email_path))

        # initial probabilities
        prob_spam = self.prob_spam
        prob_ham = self.prob_ham

        # Reduce function calculates the final values. We sum log of probabilities to
        # avoid overflow. log P(c) + summation of count(w) log P(W|c) for every word in email
        # (iterate over each word in given email)

        final_spam_probability = reduce(lambda x, word: x + self.spam_dictionary[word] * word_counter[word]
                                        if self.spam_dictionary.has_key(word)
                                        else x + self.spam_dictionary["<UNK>"] * word_counter[word],
                                        word_counter, prob_spam)

        final_ham_probability = reduce(lambda x, word: x + self.ham_dictionary[word] * word_counter[word]
                                       if self.ham_dictionary.has_key(word)
                                       else x + self.ham_dictionary["<UNK>"] * word_counter[word],
                                       word_counter, prob_ham)

        # return true if email is spam. Comparison in log space does the job (I
        # got the hint from piazza)

        return final_spam_probability > final_ham_probability

    def most_indicative_helper(self, n, dict):

        indicative_dictionary = {}

        # We initially calculated log of probabilities.
        prob_spam = math.exp(self.prob_spam)
        prob_ham = math.exp(self.prob_ham)

        for word in set(self.spam_dictionary).intersection(self.ham_dictionary):

            # numerator was already calculated.

            numerator = math.exp(dict[word])

            # p1,p2 calculated using the already computed values in init function.
            # Necessary to exponentiate to calculate denominator

            p1 = float(math.exp(self.spam_dictionary[word]) * prob_spam)
            p2 = float(math.exp(self.ham_dictionary[word]) * prob_ham)

            indicative_dictionary[word] = math.log(numerator / (p1 + p2))

        return (sorted(indicative_dictionary, key=indicative_dictionary.get, reverse=True)[:n])

    def most_indicative_spam(self, n):

        return self.most_indicative_helper(n, self.spam_dictionary)

    def most_indicative_ham(self, n):

        return self.most_indicative_helper(n, self.ham_dictionary)


class MyTest(unittest.TestCase):

    def test_accuracy(self):

        counter = 0
        errors = []

        sf = SpamFilter("data/train/spam", "data/train/ham", 1e-5)

        spam_dir = "data/dev/spam"
        ham_dir = "data/dev/ham"

        spam_files = os.listdir(spam_dir)
        ham_files = os.listdir(ham_dir)

        spam = [spam_dir + "/" + s for s in spam_files]
        ham = [ham_dir + "/" + s for s in ham_files]

        for mail in spam:
            if sf.is_spam(mail) == True:
                counter = counter + 1
            else:
                errors.append(mail)

        for mail in ham:
            if sf.is_spam(mail) == False:
                counter = counter + 1
            else:
                errors.append(mail)

        accuracy = (counter / float(len(spam) + len(ham)))

        self.assertGreaterEqual(accuracy, 0.96)


suite = unittest.TestLoader().loadTestsFromTestCase(MyTest)
unittest.TextTestRunner(verbosity=2).run(suite)
