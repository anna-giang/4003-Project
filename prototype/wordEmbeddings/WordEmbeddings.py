import string
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import os
import re
from nltk.corpus import stopwords
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


class WordEmbeddings:
    def __init__(
        self, recommendation: string,
        trainingLabelsPath: string, testLabelsPath=None,
        trainingDataPath=None, testDataPath=None
    ):

        self.currentPath = os.getcwd()
        self.recommendation = recommendation
        self.trainingLabelsPath = trainingLabelsPath
        self.testLabelsPath = testLabelsPath
        self.trainingDataPath = "prototype/" + self.recommendation + "/" + \
            "training" if trainingDataPath is None else trainingDataPath
        self.testDataPath = "prototype/" + self.recommendation + "/" + \
            "test" if testDataPath is None else testDataPath
        self.vocabulary = None
        self.model = None
        self.pretrainedModel = None
        self.train()

    def train(self) -> None:
        pathToTrainingLabelsCsv = self.currentPath + self.trainingLabelsPath

        data = self.__prepareData(
            self.trainingDataPath, pathToTrainingLabelsCsv)
        self.__loadPretrainedModel()
        print("hehe vect size", self.pretrainedModel.vector_size)
        vectorizer = Word2VecVectorizer(self.pretrainedModel)

        fittedText = vectorizer.fit_transform(data["text"])
        textFeatures = fittedText  # remove .to_array()

        # self.vocabulary = vectorizer.vocabulary_

        labels = np.array(data[self.recommendation])
        rf = RandomForestClassifier(n_estimators=200)
        self.model = rf.fit(textFeatures, labels)

    def predict(self, sample):
        if not self.model:
            self.train()
        dataFrame = pd.DataFrame()
        dataFrame["id"] = ["advertisement"]
        dataFrame["text"] = [sample]
        dataFrame["text"] = (
            dataFrame["text"]
            .apply(lambda x: Util.clean_text(x))
            .apply(lambda x: Util.remove_stopwords(x))
        )
        features = self.__vectorizeSample(dataFrame)
        return self.model.predict(features)[0]

    def assessAccuracy(self) -> None:
        """
        Prints the classification report of the accuracy of the model based on test data.
        """
        if not self.model:
            self.train()

        if not self.testLabelsPath:
            print("Please set path to test labels.")
            return

        pathToTestLabelsCsv = self.currentPath + self.testLabelsPath
        testData = self.__prepareData(self.testDataPath, pathToTestLabelsCsv)

        testTextFeatures = self.__vectorizeSample(testData)
        testLabels = np.array(testData[self.recommendation])
        print(
            metrics.classification_report(
                testLabels,
                self.model.predict(testTextFeatures),
                digits=5,
            )
        )

    def __vectorizeSample(self, data: pd.DataFrame) -> np.ndarray:
        vectorizer = Word2VecVectorizer(self.pretrainedModel)
        fittedText = vectorizer.fit_transform(data["text"])
        textFeatures = fittedText  # remove .toarray()
        return textFeatures

    def __prepareData(self, dataDirectory, pathTolabelsCsv) -> pd.DataFrame:
        filenames = []
        docs = []

        Util.loadTextFromFile(
            dataDirectory,
            filenames,
            docs,
        )

        dataFrame = pd.DataFrame()
        dataFrame["id"] = filenames
        dataFrame["text"] = docs
        dataFrame["text"] = (
            dataFrame["text"]
            .apply(lambda x: Util.clean_text(x))
            .apply(lambda x: Util.remove_stopwords(x))
        )
        labels = pd.read_csv(
            pathTolabelsCsv, converters={"id": str}
        )  # Read 'id' column as a string
        dataFrame = pd.merge(dataFrame, labels)
        return dataFrame

    def __loadPretrainedModel(self,) -> None:
        print("HER", self.currentPath)
        glove_path = self.currentPath + "/prototype" + \
            '/glove.twitter.27B.100d.txt'  # change to an argument
        word2vec_output_file = 'MY_MODEL'+'.word2vec'

        glove2word2vec(glove_path, word2vec_output_file)

        self.pretrainedModel = KeyedVectors.load_word2vec_format(
            word2vec_output_file, binary=False)

        print(self.pretrainedModel.most_similar('diversity'))


class Util:
    currentPath = os.getcwd()

    @staticmethod
    def loadTextFromFile(directory, filenames, docs):
        trainingDirectory = Util.currentPath + "/" + directory
        for filename in os.listdir(trainingDirectory):
            # Removes the .txt from the filename
            filenames.append(filename[:-4])
            with open(trainingDirectory + "/" + filename, "r") as file:
                text = file.read()
            docs.append(text)

    @staticmethod
    def clean_text(text):
        # Remove punctutation
        text = re.sub("[^a-zA-Z]", " ", text)
        # Remove numbers
        text = re.sub(r"\d+", "", text)
        # Convert to lower
        text = text.lower()
        # Remove whitespaces
        text = " ".join(text.split())
        return text

    @staticmethod
    def remove_stopwords(text: string):
        """
        Removes stopwords and short length words (< 2)
        """
        stop = set(stopwords.words("english"))
        new = []
        for word in text.split():
            if word not in stop and len(word) > 1:
                new.append(word)

        return " ".join(new)


class Word2VecVectorizer:
    def __init__(self, pretrainedModel):
        print("Loading in word vectors...")
        self.word_vectors = pretrainedModel
        print("Finished loading in word vectors")

    def fit(self, data):
        pass

    def transform(self, data):
        # determine the dimensionality of vectors
        #     v = self.word_vectors.get_vector('king')
        #     self.D = v.shape[0]
        self.D = self.word_vectors.vector_size

        X = np.zeros((len(data), self.D))
        n = 0
        emptycount = 0
        for sentence in data:
            tokens = sentence.split()
            vecs = []
            m = 0
            for word in tokens:
                try:
                    # throws KeyError if word not found
                    vec = self.word_vectors.get_vector(word)
                    vecs.append(vec)
                    m += 1
                except KeyError:
                    pass
        if len(vecs) > 0:
            vecs = np.array(vecs)
            X[n] = vecs.mean(axis=0)
        else:
            emptycount += 1
        n += 1
        print("Number of samples with no words found: %s / %s" %
              (emptycount, len(data)))
        return X

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
