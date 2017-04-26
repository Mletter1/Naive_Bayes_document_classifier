#!/usr/bin/env python
__author__ = 'Matthew Letter'
import sys
import os
import traceback
import optparse
import time
import math
import numpy
import matplotlib.pyplot


doc = """
SYNOPSIS

    main [-h,--help] [-v,--verbose] [--version] <training.txt> <validate.txt> <%confidence>

DESCRIPTION

    This is the main entry point for the ID3 python script used to create a decision tree for
    DNA. this file

EXAMPLES
    print out help message:
        python main.py -h
    Run naive bayes algorithm with verbose output
        python main.py -v training.txt validation.txt 0.95

EXIT STATUS

    0 no issues
    1 unknown error
    2 improper params

AUTHOR

    Name Matthew Letter mletter1@unm.edu

LICENSE

    This script is in the public domain, free from copyrights or restrictions.

VERSION

    v0.1
"""

"""
run the program with training and validation text files
"""


class DocumentDataStructure(object):
    """
    data structure for holding a documents info
    """

    def __init__(self, document_id=None, word_id_count_dictionary=None, label=None, predicted_class=None):
        """
        @param document_id
        @param word_id_count_dictionary dictionary mapping of id to word count
        @param label labelled id associated with document
        """
        if document_id is None or word_id_count_dictionary is None or label is None:
            raise Exception('one of the parameters to the document data structure is None')
        self.predicted_class = predicted_class
        self.document_id = document_id
        self.word_id_count_dictionary = word_id_count_dictionary
        self.label = label

    def get_sum(self, w_id):
        """
        @param w_id: this is the words w_id
        @return: count of the w_id
        """
        count = 0.0
        if w_id in self.word_id_count_dictionary:
            count = self.word_id_count_dictionary[w_id]
        return count


def standard_run():
    """
    part 1

    """
    print 'starting standard run\n'
    '''training'''
    training_docs = parse_data_files("./data/train.data", "./data/train.label")
    class_dic = classifier_learning(training_docs)

    '''testing'''
    testing_docs = parse_data_files("./data/test.data", "./data/test.label")
    get_classes(testing_docs, class_dic)

    '''results'''
    matrix = build_matrix(testing_docs)
    make_confusion_matrix(matrix)


def increment_beta():
    """increment beta and print accuracy with this variation"""
    print 'starting incremental run\n'
    beta = 0.00001
    acc_list = list()
    i_list = list()
    training_docs = parse_data_files("./data/train.data", "./data/train.label")
    testing_docs = parse_data_files("./data/test.data", "./data/test.label")
    # increment beta and store results
    while beta <= 1:
        '''need to make a copy of the list since it gets modified by the classifier'''
        temp_train_docs = list(training_docs)
        '''training'''
        class_dic = classifier_learning(temp_train_docs, beta)
        '''testing'''
        acc_list.append(get_classes(testing_docs, class_dic))
        print "i: {0}".format(beta)
        if beta >= 0.1:
            beta += 0.1
        else:
            beta *= 3
        i_list.append(beta)
    # plot the results
    plotter(acc_list, i_list)


def classify_with_top_words():
    """find the top 100 words"""
    print 'finding top 100 words\n'
    training_docs = parse_data_files("./data/train.data", "./data/train.label")
    class_dic = classifier_learning(training_docs)
    vocab_list = parse_vocab()  # optionally you can pass your own file path here
    common_words = parse_vocab("./data/commonwords.txt")
    words = get_top_words(class_dic, vocab_list, common_words)
    print 'classifying with top 100 words\n'
    testing_docs = parse_data_files("./data/test.data", "./data/test.label")
    get_classes(testing_docs, class_dic, words)
    '''results'''
    matrix = build_matrix(testing_docs)
    make_confusion_matrix(matrix)


def parse_vocab(file_url="./data/vocabulary.txt"):
    """parse vocabulary file"""
    vocab_file = open(file_url, "r+")
    vocab_list = list()
    for line in vocab_file:
        line = line.rstrip('\n')
        if line not in vocab_list:
            vocab_list.append(line)
    return vocab_list


def plotter(x=None, y=None):
    """for plotting general x->y with a log scale"""
    if x is None or y is None:
        raise Exception('one of the parameters to the document data structure is None')
    for i in range(len(x)):
        print x[i], y[i]
    matplotlib.pyplot.xscale('log')
    matplotlib.pyplot.plot(y, x, 'k')
    matplotlib.pyplot.show()


def build_matrix(testing_docs):
    """
    initialize a 20X20 matrix and add doc count for each slot
    :parameter testing_docs list of docs used for making matrix
    """
    matrix = numpy.zeros((20, 20))  # a matrix, first dimension is actual value, second dimension is predict value
    for document in testing_docs:
        matrix[int(document.label) - 1][int(document.predicted_class) - 1] += 1
    return matrix


def make_confusion_matrix(matrix):
    """
    take a matrix and use matplotlib to create a confusion matrix graph
    :parameter matrix word count matrix
    """
    count = []
    ticks = []
    # create tick marks and calculate each sqaure's value in the matrix
    for k in range(20):
        count.append(sum(matrix[k]))
        ticks.append(k + 1)
        for j in range(20):
            matrix[k][j] = (matrix[k][j] / (count[k])) * 100
        matplotlib.pyplot.imshow(matrix, aspect='auto')
    # add annotations
    for k in range(20):
        for j in range(20):
            point = str(round(matrix[k][j], 0)).replace(".0", "")
            matplotlib.pyplot.annotate(point, xy=(j, k), horizontalalignment='center')
    # settup the plot info
    matplotlib.pyplot.xlabel('Predicted')
    matplotlib.pyplot.ylabel('Actual')
    matplotlib.pyplot.title("Confusion matrix (Values are %)")
    matplotlib.pyplot.xticks(range(20), ticks)
    matplotlib.pyplot.yticks(range(20), ticks)
    matplotlib.pyplot.show()


def get_top_words(class_dictionary, vocab_list, common_words):
    """find the top 100 words given vocab list and class dictionary"""
    words_dictionary = {}
    standard_deviation_list = list()
    num = len(vocab_list)
    #go through every vocabulary word
    #http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.80.2934&rep=rep1&type=pdf
    for i in range(num):
        # remove common words
        if vocab_list[i] not in common_words:
            words_list = list()
            ss = 0.0
            for j in range(20):
                if str(i + 1) in class_dictionary[str(j + 1)]:
                    words_list.append(class_dictionary[str(j + 1)].get(str(i + 1)))
                else:
                    words_list.append(0)
            mean = sum(words_list) / len(words_list)
            for item in words_list:
                ss += math.pow((item - mean), 2)
            ss /= len(words_list)
            words_dictionary[str(math.sqrt(ss))] = i + 1
            standard_deviation_list.append(math.sqrt(ss))
    standard_deviation_list.sort()
    top_word_list = list()
    for i in range(100):
        top_word_list.append(words_dictionary[str(standard_deviation_list[len(standard_deviation_list) - i - 1])])
        print vocab_list[top_word_list[i]], ",",
    print ""
    return top_word_list


def get_classes(document_list=None, classification_dictionary=None, words=None):
    """
    This method is called to classify testing samples.
    :parameter document_list list of docs to classify
    :parameter classification_dictionary this is our classifier
    """
    correct = 0
    for document in document_list:  # iterate each test doc.
        max_p = float("-inf")
        for current_class in classification_dictionary.keys():  # iterate each newsgroup to find the one with highest p.
            current_p = math.log((classification_dictionary[current_class])["pc"])
            if words is None:
                for word_id in document.word_id_count_dictionary.keys():
                    if word_id in classification_dictionary[current_class]:
                        current_words_probability = (classification_dictionary[current_class])[word_id]
                    else:
                        current_words_probability = classification_dictionary[current_class]["d"]
                    # summing probability
                    current_p += (math.log(current_words_probability) * document.word_id_count_dictionary[word_id])
            else:
                for word_id in words:
                    word_id_use = str(word_id)
                    if word_id_use in classification_dictionary[current_class]:
                        current_words_probability = (classification_dictionary[current_class])[word_id_use]
                    else:
                        current_words_probability = classification_dictionary[current_class]["d"]
                        # summing probability
                    current_p += (math.log(current_words_probability) * document.get_sum(word_id_use))
            # P(Y|X) = agrmax log(P(Yk))+ sum log(P(Xi))
            if current_p > max_p:
                max_p = current_p
                document.predicted_class = current_class
        if document.predicted_class == document.label:
            correct += 1
    accuracy = (1.0 * correct / (len(document_list)) * 100)
    print "Accuracy: ", accuracy
    return accuracy


def classifier_learning(document_list=list(), a=None):
    """
    train a classifier
    :parameter document_list list of ducments for training
    :parameter a is the beta that could potentially be varied
    here is the idea
    https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0CB4QFjAA&url=https%3A%2F%2Fweb.
    stanford.edu%2Fclass%2Fcs124%2Flec%2Fnaivebayes.pptx&ei=nDsKVe-iMoauogTF1IHABw&usg=AFQjCNGhk2UgGbj1IL
    E7SEis5f9w4ldCrA&sig2=OplBTNyVpVcat5Nx6eFw1A&bvm=bv.88528373,d.cGU
    """
    classification_dictionary = {}
    # calculate a where a = 1/|V| if no given a
    if a is None:
        count = 0.0
        for _ in open("./data/vocabulary.txt", "r+"):
            count += 1.0
        a = (1.0 / count)

    # find word document info
    for document in document_list:
        if document.label not in classification_dictionary:  # initialize the map
            classification_dictionary[document.label] = {"pc": 0, "wc": 0, "a": a}
        probability_map = classification_dictionary[document.label]
        probability_map["pc"] += 1
        for key in document.word_id_count_dictionary.keys():
            if key in probability_map:
                probability_map[key] += document.word_id_count_dictionary[key]
            else:
                probability_map[key] = document.word_id_count_dictionary[key]
            probability_map["wc"] += document.word_id_count_dictionary[key]

    # find P(w|c) with MAP and find P(c) with MLE.
    for probability_map in classification_dictionary.values():
        for key in probability_map.keys():
            if key is not "pc" and key is not "wc":
                # P(w|c)
                probability_map[key] = ((probability_map[key] + a) / probability_map["wc"])
            elif key is "pc":
                # P(class)
                probability_map[key] = 1.0 * probability_map[key] / len(document_list)
        # probability_map["d"] = 0.00001
        probability_map["d"] = a / probability_map["wc"]
    return classification_dictionary


def parse_data_files(data_file_path, label_file_path):
    """
    takes a file of data and label and creates a list of
    DocumentDataStructure's that represent the combined
    :parameter data_file_path url to data file
    :parameter label_file_path url to label data file
    """
    # get labels and inject them into a list
    labels = list()
    for line in open(label_file_path, "r+"):
        line = line.rstrip('\n')
        labels.append(line)

    # setup for building out the data
    data_reader = open(data_file_path, "r+")
    document_list = list()
    word_id_count_dictionary = {}
    line = data_reader.readline().rstrip('\n')
    doc_elements = line.split()
    document_id = int(doc_elements[0])

    # get data and combine with labels
    for line in data_reader:
        line = line.rstrip('\n')
        doc_elements = line.split()
        if int(doc_elements[0]) != document_id:
            label_index = (document_id - 1)
            new_doc = DocumentDataStructure(int(document_id), word_id_count_dictionary, labels[label_index])
            document_list.append(new_doc)
            word_id_count_dictionary = {}
        document_id = int(doc_elements[0])
        word_id_count_dictionary[doc_elements[1]] = int(doc_elements[2])  # map word_id to count
    label_index = (int(document_id) - 1)
    document_list.append(DocumentDataStructure(int(document_id), word_id_count_dictionary, labels[label_index]))
    return document_list


def run():
    print "\n" + "***********************************"
    print 'Runnning main script\n'
    standard_run()
    increment_beta()
    classify_with_top_words()
    print "***********************************\n"


def run_quite():
    standard_run()
    increment_beta()
    classify_with_top_words()


"""
determine running params
"""
if __name__ == '__main__':
    global options, args
    try:
        start_time = time.time()
        parser = optparse.OptionParser(formatter=optparse.TitledHelpFormatter(), usage=doc,
                                       version='%prog 0.1')

        parser.add_option('-v', '--verbose', action='store_true', default=False, help='verbose output')
        # get the options and args
        (options, args) = parser.parse_args()

        # determine what to do with the options supplied by the user
        if options.verbose:
            print "options ", options
            print "args", args
            print "start time: " + time.asctime()
            run()
            print "finish time: " + time.asctime()
            print 'TOTAL TIME IN MINUTES:',
            print (time.time() - start_time) / 60.0
        else:
            run_quite()
        # smooth exit if no exceptions are thrown
        sys.exit(0)

    except KeyboardInterrupt, e:  # Ctrl-C
        raise e
    except SystemExit, e:  # sys.exit()
        raise e
    except Exception, e:  # unknown exception
        print 'ERROR, UNEXPECTED EXCEPTION'
        print str(e)
        traceback.print_exc()
        os._exit(1)