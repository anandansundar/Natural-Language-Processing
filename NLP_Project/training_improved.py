import wikipedia
from textblob.classifiers import NaiveBayesClassifier
import csv
import nltk
from nltk.tokenize import MWETokenizer, word_tokenize
from nltk.corpus import stopwords
import pickle
from nltk import pos_tag
from textblob import Word
from nltk import RegexpParser
from nltk.corpus import wordnet

train = []

reader = csv.reader(open('Topic_set_train.csv', 'r'))
tokenizer = MWETokenizer()


for row in reader:
    print("Data : " + str(row))
    title, category = row
    tokenizer.add_mwe(title.split())
    wiki_page = wikipedia.page(title)
    wiki_content = str.lower(wiki_page.summary)
    tokens = tokenizer.tokenize(wiki_content.split())
    tokens_content = " ".join(tokens)
    word_tokens = word_tokenize(tokens_content)
    bigger_words = [k for k in word_tokens if len(k) >= 3 and not k.startswith('===')]
    stop = set(stopwords.words('english'))
    stopwords_cleaned_list = [k for k in bigger_words if k not in stop]
    lemmatized_tokens = []

    for word in stopwords_cleaned_list:
        w = Word(word)
        lemmatized_tokens.append(w.lemmatize())

    # print(lemmatized_tokens)

    pos_tagged_word_list = pos_tag(lemmatized_tokens)

    grammar = """ NP: {<DT>?<JJ>*<NN>}
                      {<NNP>+}
                      {<NN><NN>}
                      {<NNS><VBP>}
                      {<V.*> <TO> <V.*>}
                      {<N.*>(4,)} """

    NPChunker = RegexpParser(grammar)
    chunked_result = NPChunker.parse(pos_tagged_word_list)

    # print(chunked_result)
    shallow_parsed_list = list()

    for sub_tree in chunked_result:
        if type(sub_tree) is nltk.tree.Tree:
            if sub_tree.label() == 'NP':
                for w, t in sub_tree.leaves():
                    if 'NN' in t:
                        shallow_parsed_list.append(w)

    # print(shallow_parsed_set)

    hypernym_parsed_set = list()
    meronym_parsed_set = list()

    for text in shallow_parsed_list:
        for synset in wordnet.synsets(text, pos='n'):
            word = synset.name()
            word_synset = wordnet.synset(word)
            # Accessing hypernyms
            hypernym_list = word_synset.hypernyms()
            for hypernym in hypernym_list:
                if "business" in text or "technology" in text or "politics" in text or "travel" in text:
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

    #print("After Hypernym matching:")
    #print(hypernym_parsed_set)

    for text in hypernym_parsed_set:
        for synset in wordnet.synsets(text, pos='n'):
            word = synset.name()
            word_synset = wordnet.synset(word)
            # Accessing hypernyms
            meronym_list = word_synset.part_meronyms()
            for meronym in meronym_list:
                if "business" in text or "technology" in text or "politics" in text or "travel" in text:
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

    text_content = " ".join(meronym_parsed_set)
    train_example = (text_content, category)
    train.append(train_example)


print("Building Model")
classifier = NaiveBayesClassifier(train)
save_classifier = open("naivebayes_improved4.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()