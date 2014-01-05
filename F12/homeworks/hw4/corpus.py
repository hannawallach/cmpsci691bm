import cPickle as pickle

from vocabulary import *


class Document(object):

    def __init__(self, corpus, name, w):

        self.corpus = corpus
        self.name = name
        self.w = w

    def __len__(self):
        return len(self.w)

    def plaintext(self):
        return ' '.join([self.corpus.vocab.lookup(x) for x in self.w])


class Corpus(object):

    def __init__(self):

        self.documents = []
        self.vocab = Vocabulary()

    def add(self, name, tokens):

        w = [self.vocab[x] for x in tokens]
        self.documents.append(Document(self, name, w))

    def __iter__(self):
        return iter(self.documents)

    def __len__(self):
        return len(self.documents)

    @classmethod
    def load(cls, filename):
        return pickle.load(file(filename, 'r'))

    def save(self, filename):
        pickle.dump(self, file(filename, 'wb'))


class GroupedCorpus(Corpus):

    def __init__(self):

        self.z = []
        self.group_vocab = Vocabulary()
        Corpus.__init__(self)

    def add(self, name, group, tokens):

        self.z.append(self.group_vocab[group])
        Corpus.add(self, name, tokens)

    def __iter__(self):
        return iter(zip(self.documents, self.z))
