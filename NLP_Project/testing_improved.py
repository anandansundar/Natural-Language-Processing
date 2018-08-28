import wikipedia
import csv
import nltk
from nltk.tokenize import MWETokenizer, word_tokenize
from nltk.corpus import stopwords
import pickle
from nltk import pos_tag
from textblob import Word
from nltk import RegexpParser
from nltk.corpus import wordnet

test = []
title_name = []
reader = csv.reader(open('Topic_set_test.csv', 'r'))
tokenizer = MWETokenizer()


def equating(content):
    if "technology" in content:
        content = "technology"
    if "business" in content:
        content = "business"
    if "politics" in content:
        content = "politics"
    if "travel" in content:
        content = "travel"
    return content


for row in reader:
    print("Data : " + str(row))
    title, category = row
    tokenizer.add_mwe(title.split())
    wiki_page = wikipedia.page(title)
    title_name.append(title)
    wiki_content = str.lower(wiki_page.summary)
    tokens = tokenizer.tokenize(wiki_content.split())
    tokens_content = " ".join(tokens)
    word_tokens = word_tokenize(tokens_content)
    bigger_words = [k for k in word_tokens if len(k) >= 3 and not k.startswith('===')]
    stop = set(stopwords.words('english'))
    stopwords_cleaned_list = [k for k in bigger_words if k not in stop]
    lemmatized_tokens = []

    print("After tokenizing and removing stopwords : ")
    print(stopwords_cleaned_list)

    for word in stopwords_cleaned_list:
        w = Word(word)
        lemmatized_tokens.append(w.lemmatize())

    print("After Lemmatization :")
    print(lemmatized_tokens)

    pos_tagged_word_list = pos_tag(lemmatized_tokens)
    print("After POS Tagging :")
    print(pos_tagged_word_list)

    grammar = """ NP: {<DT>?<JJ>*<NN>}
                      {<NNP>+}
                      {<NN><NN>}
                      {<NNS><VBP>}
                      {<V.*> <TO> <V.*>}
                      {<N.*>(4,)} """

    NPChunker = RegexpParser(grammar)
    chunked_result = NPChunker.parse(pos_tagged_word_list)
    shallow_parsed_set = list()

    for sub_tree in chunked_result:
        if type(sub_tree) is nltk.tree.Tree:
            if sub_tree.label() == 'NP':
                for w, t in sub_tree.leaves():
                    if 'NN' in t:
                        shallow_parsed_set.append(w)

    print("After Chunking (Shallow Parsing) :")
    print(shallow_parsed_set)

    hypernym_parsed_set = list()
    meronym_parsed_set = list()

    for text in shallow_parsed_set:
        for synset in wordnet.synsets(text, pos='n'):
            word = synset.name()
            word_synset = wordnet.synset(word)
            # Accessing hypernyms
            hypernym_list = word_synset.hypernyms()
            for hypernym in hypernym_list:
                if "business" in text or "technology" in text or "politics" in text or "travel" in text:
                    print("Topic has been found in hypernym! " + text)
                    isChanged = False
                    if "technology" in text:
                        text = "technology"
                        isChanged = True
                    if "business" in text:
                        text = "business"
                        isChanged = True
                    if "politics" in text:
                        text = "politics"
                        isChanged = True
                    if "travel" in text:
                        text = "travel"
                        isChanged = True
                    if isChanged:
                        hypernym_parsed_set.append(text)
        hypernym_parsed_set.append(text)

    print("After Hypernym :")
    print(hypernym_parsed_set)

    for text in hypernym_parsed_set:
        for synset in wordnet.synsets(text, pos='n'):
            word = synset.name()
            word_synset = wordnet.synset(word)
            # Accessing hypernyms
            meronym_list = word_synset.part_meronyms()
            for meronym in meronym_list:
                if "business" in text or "technology" in text or "politics" in text or "travel" in text:
                    print("Topic has been found in meronym! " + text)
                    isChanged = False
                    if "technology" in text:
                        text = "technology"
                        isChanged = True
                    if "business" in text:
                        text = "business"
                        isChanged = True
                    if "politics" in text:
                        text = "politics"
                        isChanged = True
                    if "travel" in text:
                        text = "travel"
                        isChanged = True
                    if isChanged:
                        meronym_parsed_set.append(text)
        meronym_parsed_set.append(text)

    print("After Meronym matching : ")
    print(meronym_parsed_set)

    text_content = " ".join(meronym_parsed_set)
    test_example = (text_content, category)
    test.append(test_example)

classifier_f = open("naivebayes_improved4.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

i = 0
for test_sample in test:
    classified = classifier.classify(test_sample[0])
    print("Topic : " + title_name[i])
    print("Actual : " + str(test_sample[1]).strip())
    print("Classified : " + classified)
    i += 1

print("Accuracy : " + str(classifier.accuracy(test) * 100))
