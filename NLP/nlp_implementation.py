import nltk
from nltk.stem.lancaster import LancasterStemmer

import numpy as np
import random
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model

import json


class NLPImplementation:
    def __init__(self, intents_location):
        self.intents_ignore_words = ["?", "!", ".", ","]
        self.ERROR_THRESHOLD = 0.25
        self.model = None
        self.intents_location = intents_location
        self.stemmer = LancasterStemmer()
        self.intents_words, self.intents_documents, self.intents_classes = self.apply_tokenization_on_intents()
        self.model_save_name = "chatbot_model.h5"

    def clean_up_sentence(self, sentence):
        # tokenize the pattern
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word
        sentence_words = [self.stemmer.stem(word.lower()) for word in sentence_words]

        return sentence_words

    def bag_of_words(self, sentence):
        """ return bag of words array: 0 or 1 for each word in the bag that exists in the sentence"""
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words
        bag = [0] * len(self.intents_words)
        for sw in sentence_words:
            for index, word in enumerate(self.intents_words):
                if word == sw:
                    bag[index] = 1

        return np.array(bag)

    def response(self, sentence):
        with open(self.intents_location) as json_data:
            intents = json.load(json_data)
            json_data.close()

        results = self.classify(sentence)
        # if classification exists then find the matching intent tag and return a response from the respective tag
        if results:
            # loop as long as there are matches to process
            while results:
                for i in intents['intents']:
                    # find a tag matching the first result
                    if i['tag'] == results[0]["intent"]:
                        # return a random response from the intent
                        return random.choice(i['response'])

                results.pop(0)

    def classify(self, sentence):
        # generate probabilities from the model
        self.load_model()

        bow = self.bag_of_words(sentence)
        results = self.model.predict(np.array([bow]))[0]

        # Filters out predictions below a threshold
        results = [[i, res] for i, res in enumerate(results) if res > self.ERROR_THRESHOLD]

        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.intents_classes[r[0]], "probability": r[1]})

        # return dict of intent and probability
        print(return_list)
        return return_list

    def apply_tokenization_on_intents(self):
        documents = []
        words = []
        classes = []

        with open(self.intents_location) as json_data:
            intents = json.load(json_data)
            json_data.close()

        for intent in intents["intents"]:
            for pattern in intent["patterns"]:
                #  Tokenize each word
                word = nltk.word_tokenize(pattern)
                words.extend(word)

                # Add to documents in our corpus
                documents.append((word, intent["tag"]))

                # Add to classes list
                if intent["tag"] not in classes:
                    classes.append(intent["tag"])

        words = [self.stemmer.stem(w.lower()) for w in words if w not in self.intents_ignore_words]
        words = sorted(list(set(words)))

        # Removes duplicates
        classes = sorted(list(set(classes)))

        # print(f"Document Length: {len(documents)}")
        # print(f"Classes length: {len(classes)} contains: \n {classes}")
        # print(f"Number of unique stemmed words: {len(words)} contains: \n {words}")

        return words, documents, classes

    def create_training_data(self):
        training = []
        # create an empty array for our output
        output_empty = [0] * len(self.intents_classes)

        # training set, bag of words for each sentence
        for doc in self.intents_documents:
            # initialize our bag of words
            bag = []

            # list of tokenized words for the pattern
            pattern_words = doc[0]

            # stem each word
            pattern_words = [self.stemmer.stem(word.lower()) for word in pattern_words]

            # create our bag of words array
            for word in self.intents_words:
                bag.append(1) if word in pattern_words else bag.append(0)

            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[self.intents_classes.index(doc[1])] = 1

            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training)

        train_x = list(training[:, 0])
        train_y = list(training[:, 1])

        return [train_x, train_y]

    def train_model(self):
        # Build neural network

        train_x, train_y = self.create_training_data()
        model = Sequential()
        model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(train_y[0]), activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

        model_fit = model.fit(np.array(train_x), np.array(train_y), epochs=2000, batch_size=5, verbose=1)

        model.save(self.model_save_name, model_fit)
        print("Training Complete")

        pickle.dump(
            {
                'words': self.intents_words,
                'classes': self.intents_classes,
                'train_x': train_x,
                'train_y': train_y},
            open("training_data", "wb"),
        )

    def load_model(self):
        """Makes sure that self.model is loaded to be used for predictions"""
        try:
            data = pickle.load(open("training_data", "rb"))
            words = data['words']
            classes = data['classes']
            train_x = data['train_x']
            train_y = data['train_y']

            self.model = load_model(self.model_save_name)
        except FileNotFoundError as e:
            print("Model was not trained. Now training model")
            self.train_model()
            self.model = load_model(self.model_save_name)
