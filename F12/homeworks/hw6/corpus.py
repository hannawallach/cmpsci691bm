import cPickle as pickle
from numpy import bincount, zeros

from vocabulary import Vocabulary


class Document(object):

    def __init__(self, corpus, name, w):

        self.corpus = corpus
        self.name = name
        self.w = w

        self.Nv = None

    def __len__(self):
        return len(self.w)

    def freeze(self):

        self.Nv = zeros(len(self.corpus.vocab), dtype=int)
        self.Nv[:max(self.w) + 1] = bincount(self.w)

    def plaintext(self):
        return ' '.join([self.corpus.vocab.lookup(x) for x in self.w])


class Corpus(object):

    def __init__(self, documents=None, vocab=None, frozen=None):

        if documents:
            self.documents = documents
        else:
            self.documents = []

        if vocab:
            self.vocab = vocab
        else:
            self.vocab = Vocabulary()

        if frozen:
            self.frozen = frozen
        else:
            self.frozen = False

    def add(self, name, tokens):

        if not self.frozen:
            w = [self.vocab[x] for x in tokens]
            self.documents.append(Document(self, name, w))

    def freeze(self):

        for doc in self.documents:
            doc.freeze()

        self.vocab.stop_growth()
        self.frozen = True

    def __getitem__(self, i):
        return self.documents[i]

    def __getslice__(self, i, j):
        return Corpus(self.documents[i:j], self.vocab, self.frozen)

    def __iter__(self):
        return iter(self.documents)

    def __len__(self):
        return len(self.documents)

    @classmethod
    def load(cls, filename):
        return pickle.load(file(filename, 'r'))

    def save(self, filename):
        pickle.dump(self, file(filename, 'wb'))
