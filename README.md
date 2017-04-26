Matthew Letter
March 19 2015
mletter1@unm.edu

Overview:

The overall goal of the code was to use Naïve Bayes for classifying documents.
The first part of which was training a Bayesian system. The data.txt and label.txt
file were first parsed and turned into document objects and stored in a list. This
list was used to train the Bayesian classifier.  The document list was passed on and
the probability of each word for each topic was calculated. Beta was set to 1/|vocabulary|.
The reverted value for any word not found in a category was set to beta/|words in category|.
P(xi|yj) was set to |word in category|+beta/|words in category| and P(y) was set to |documents
in category/|number of documents|, and P(y|x) was obtained by taking the max value of the
log of P(yj) and adding the sum of all log P(xi). Which ever had the highest value was
used as the classifier for the category. After the classifier was trained the test documents
were loaded and parsed using the same approach as the training data. The test values were
then classified using the trained Bayesian classifier. The number of correctly and
incorrectly classified documents was recorded down and used to determine the % accuracy.
The accuracy was the number of correct over |documents|.

Running the code:

run the code:
python main.py

runt the code in verbose mode:
python main.py -v
or
pythong main.py -verbose

help:
python main.py -h
or
python main.py -help

the code will fist run with beta = 1/|Vocabulary|  and print out a confusion matrix.  the code will
then do an incremental run with beta from 0.00001 to 2 and print out a graph with an accuracy result
lastly the top 100 words will be printed out and then be used to classify test documents.