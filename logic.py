import pickle
import json
from nltk.stem.lancaster import LancasterStemmer
import random
import tensorflow as tf
import tflearn
import numpy
import nltk
nltk.download('punkt')
data = json.loads('''{"intents": [
        {"tag": "greeting",
         "patterns": ["Hi", "How are you", "Is anyone there?", "Hello", "Good day", "Whats up"],
         "responses": ["Hello!", "Good to see you again!", "Hi there, how can I help?"],
         "context_set": ""
        },
        {"tag": "goodbye",
         "patterns": ["cya", "See you later", "Goodbye", "so long!", "don't let the bed bugs rape!"],
         "responses": ["Sad to see you go :(", "Talk to you later", "Goodbye!","Good riddance!"],
         "context_set": ""
        },
        {"tag": "age",
         "patterns": ["how old", "how old is tim", "what is your age", "how old are you", "age?"],
         "responses": ["I am t-2 years old!", "t-2 years young!"],
         "context_set": ""
        },
        {"tag": "name",
         "patterns": ["what is your name", "what should I call you", "whats your name?"],
         "responses": ["You can call me Trusty.", "I'm Trusty!", "I'm Trusty aka Trusty the Chatbot."],
         "context_set": ""
        },
        {"tag": "shop",
         "patterns": ["Id like to buy something", "whats on the menu", "what do you reccommend?", "could i get something to drink"],
         "responses": ["We sell lemonade for $2!", "Lemonade is on the menu!"],
         "context_set": ""
        },
        {"tag": "hours",
         "patterns": ["when are you guys open", "what are your hours", "hours of operation"],
         "responses": ["We are open 7am-4pm Monday-Friday!"],
         "context_set": ""
        }
   ]
}''')
stemmer = LancasterStemmer()
nltk.download('punkt')

try:
    s
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

try:
    model.load("model.tflearn")
    print("loaded from file")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)


def call(what):
    imp = input("You: ")
    results = model.predict([bag_of_words(imp, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    for tg in data['intents']:
        if tg['tag'] == tag:
            responses = tg['responses']
    print(random.choice(responses))
