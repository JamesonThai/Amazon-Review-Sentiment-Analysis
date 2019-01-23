import json
import re
import string
import numpy
import random

from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

class AmazonDataCleaner:

    char_table = str.maketrans('', '', string.punctuation)

    def __init__(self, json_files):
        self.json_files = json_files
        self.data_files = []

        self.reviews = { 1 : [], 2 : [], 3 : [], 4 : [], 5 : [] }

    def run(self):
        for filename in self.json_files:
            self.parse_file(filename)
        self.write_files()

    def parse_file(self, filename):
        with open(filename) as f:
            for line in f:
                json_obj = json.loads(line)
                rating = int(json_obj['overall'])
                self.reviews[rating].append(AmazonDataCleaner.format_text(json_obj['reviewText']))

    def write_files(self):
        for key, review in self.reviews.items():
            filename = '%s%d%s' % ('./reviews/reviews', key, '.txt')
            self.data_files.append(filename)
            with open(filename, 'w') as f:
                for r in review:
                    f.write('%s\n' % r)

    def get_files(self):
        return self.data_files

    @staticmethod
    def format_text(review):
        return review.translate(AmazonDataCleaner.char_table).lower()


class DataPartitioner:
    def __init__(self, data_files, training_total, testing_total):
        self.DATA_FILES = data_files
        self.TRAINING_TOTAL = training_total
        self.TESTING_TOTAL = testing_total 
        self.TRAINING_DIRECTORY = './training_data'
        self.TESTING_DIRECTORY = './testing_data'

        self.labeled_output = dict()

    
    def run(self):
        for i in range(1, len(self.DATA_FILES)+1):
            with open(self.DATA_FILES[i-1]) as f:
                lines = f.readlines()
            
            random.shuffle(lines)
            tr_filename = self.TRAINING_DIRECTORY + '/reviews' + str(i) + '.txt'
            self.labeled_output[tr_filename] = str(i) + 'star_TR'
            with open(tr_filename, 'w') as f:
                f.write(''.join(lines[0:self.TRAINING_TOTAL]))

            te_filename = self.TESTING_DIRECTORY + '/reviews' + str(i) + '.txt'
            self.labeled_output[te_filename] = str(i) + 'star_TE'
            with open(te_filename, 'w') as f:
                f.write(''.join(lines[self.TRAINING_TOTAL:self.TRAINING_TOTAL+self.TESTING_TOTAL]))

    def get_labeled_output(self):
        return self.labeled_output



# Title: word2vec-sentiment.ipynb
# Author: Linan Qiu
# Date: 4/24/2018
# Availability: https://github.com/linanqiu/word2vec-sentiments/blob/master/word2vec-sentiment.ipynb
class LabeledLineSentence:
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
    
    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
    
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences
    
    def sentences_perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return shuffled


# Code inspired by Linan Qiu
class Doc2VecTrainer:
    def __init__(self, sources, **kwargs):
        self.sentences = LabeledLineSentence(sources)
        self.model = Doc2Vec(**kwargs)
        self.model.build_vocab(self.sentences.to_array())

    def train(self, epoch_count):
        for e in range(epoch_count):
            print('Training round %d out of %d' % (e + 1, epoch_count))
            self.model.train(self.sentences.sentences_perm(),
                             epochs=self.model.iter,
                             total_examples=self.model.corpus_count)

    def save(self, filename):
        self.model.save(filename + '.d2v')


class ModelTrainer:

    def __init__(self, model, training_count, testing_count):
        self.model = model

        self.TRAIN_COUNT = training_count
        self.TEST_COUNT = testing_count

        self.RATING_LOW = 1
        self.RATING_HIGH = 5
        self.TR_SET_STR = '%sstar_TR_%s'
        self.TE_SET_STR = '%sstar_TE_%s'
        
        self.classifier = None


    def generate_sets(self, rating_range=None):
        
        if rating_range == None:
            rating_range = range(self.RATING_LOW, self.RATING_HIGH + 1)

        self.train_x = numpy.zeros((self.TRAIN_COUNT*len(rating_range), self.model.vector_size))
        self.train_y = numpy.zeros(self.TRAIN_COUNT*len(rating_range))
        self.test_x = numpy.zeros((self.TEST_COUNT*len(rating_range), self.model.vector_size))
        self.test_y = numpy.zeros(self.TEST_COUNT*len(rating_range))

        for idx in range(len(rating_range)):
            i = rating_range[idx]
            for j in range(self.TRAIN_COUNT):
                self.train_x[(idx)*self.TRAIN_COUNT + j] = self.model[self.TR_SET_STR % (i, j)]
                self.train_y[(idx)*self.TRAIN_COUNT + j] = i
            for j in range(self.TEST_COUNT):
                self.test_x[(idx)*self.TEST_COUNT + j] = self.model[self.TE_SET_STR % (i, j)]
                self.test_y[(idx)*self.TEST_COUNT + j] = i


    def train(self, classifier, ratio=1):
        if ratio > 1:
            ratio = 1
        self.classifier = classifier
        self.classifier.fit(self.train_x[0:int(len(self.train_x)*ratio)], self.train_y[0:int(len(self.train_y)*ratio)])
        return self.classifier

    def score(self, ratio=1):
        if ratio > 1:
            ratio = 1
        return self.classifier.score(self.test_x[0: int(len(self.test_x)*ratio)], self.test_y[0: int(len(self.test_y)*ratio)])

    def classify(self, review):
        vec = self.model.infer_vector(AmazonDataCleaner.format_text(review).split())
        return self.classifier.predict([vec])


# Example flow of execution. This will take an hour or two to run.
'''
if __name__ == '__main__':
    AMAZON_JSON_FILES = [
        './dataset/reviews_Electronics_5.json',
        './dataset/reviews_Books_5.json',
        './dataset/reviews_Movies_and_TV_5.json'
    ]
    TRAINING_TOTAL = 10000
    TESTING_TOTAL = 5000

    json_cleaner = AmazonDataCleaner(AMAZON_JSON_FILES)
    json_cleaner.run()

    partitioner = DataPartitioner(json_cleaner.get_files(), TRAINING_TOTAL, TESTING_TOTAL)
    partitioner.run()

    trainer = Trainer(partitioner.get_labeled_output(),
        min_count=1, window=5, vector_size=400, sample=1e-4, negative=5, workers=4
    )

    trainer.train(10)
    trainer.save('amazon_data')

    model_trainer = ModelTrainer(trainer.model, TRAINING_TOTAL, TESTING_TOTAL)
    model_trainer.generate_sets()
    
    clf = LogisticRegression()
    model_trainer.train(clf)
    print(model_trainer.score())

'''
