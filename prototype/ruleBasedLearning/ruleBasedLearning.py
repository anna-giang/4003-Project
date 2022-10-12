import re
import os.path
masculine = open(os.path.dirname(__file__) +
                 '/data/dict/masculine_dict.txt').read().splitlines()
feminine = open(os.path.dirname(__file__) +
                '/data/dict/feminine_dict.txt').read().splitlines()
target1 = open(os.path.dirname(__file__) +
               '/data/dict/target1_dict.txt').read().splitlines()
target2 = open(os.path.dirname(__file__) +
               '/data/dict/target2_dict.txt').read().splitlines()
target3 = open(os.path.dirname(__file__) +
               '/data/dict/target3_dict.txt').read().splitlines()
target5 = open(os.path.dirname(__file__) +
               '/data/dict/target5_dict.txt').read().splitlines()
target6 = open(os.path.dirname(__file__) +
               '/data/dict/target6_dict.txt').read().splitlines()
target7 = open(os.path.dirname(__file__) +
               '/data/dict/target7_dict.txt').read().splitlines()
stopwords = open(os.path.dirname(__file__) +
                 '/data/stopwords.txt').read().splitlines()


def count(text, words):
    n = 0
    for word in words:
        if word in text:
            n += 1
    return n


def process(text):

    # lowers case
    text = text.lower()

    # remove numbers
    text = re.sub(r'\d+', '', text)

    technical = [x for x in text if x not in stopwords]
    # 0 - number of diverisity/inclusion
    # 1 - number of encourage gender]
    # 2 - number of feature or organistion/team
    # 3 - number of attributes tagged male/female * not sure about this
    # 4 - work hour features such as flexible/strict work hours, weekend hours:
    # 5 - number of type of the job mention
    # 6 - number of masculine words
    # 7 - numbre of feminine words
    # 8 - number of technical words used
    results = [count(text, target1),
               count(text, target2),
               count(text, target3),
               count(text, target5),
               count(text, target6),
               count(text, target7),
               count(text, masculine),
               count(text, feminine),
               len(technical)]
    return [results[4], results[3], results[5], results[6], results[7], results[8]]
