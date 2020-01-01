import re
import string
import numpy as np
import pandas as pd
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from gensim import  models
from gensim.utils import simple_preprocess
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

medium = pd.read_csv(r'C:\Users\Riya Banerjee\PycharmProjects\content_recommender_demo\medium-articles-with-content (1)/true_Medium_AggregatedData.csv')
# removing articles that are not in english
medium = medium[medium['language'] == 'en']
# removing articles which have low rating
medium = medium[medium['totalClapCount'] >= 25]
# in order to decrease runtime
#medium= medium.head(50)

def findTags(title):
    rows = medium[medium['title'] == title]
    # getting the tags for a particular title and storing it in a list for convenience
    tags = list(rows['tag_name'])
    return tags

def addTags(title):
    try:
        tags = list(tag_df[tag_df['title'] == title]['tags'])[0]
    except:
        #if there are no tags for a title
        tags=np.NaN
    return tags

# define a tag_dict for relating all the tags with their title
tag_dict= {'title' : [], 'tags' : []}

# to fill the tag_dict
titles= medium['title'].unique()
# to obtain title and tags of each article
for title in titles:
    tag_dict['title'].append(title)
    tag_dict['tags'].append(findTags(title))
tag_df= pd.DataFrame(tag_dict)

# since we have obtained all tags, we can remove duplicate titles
medium = medium.drop_duplicates(subset='title', keep='first')

# add a column allTags to insert all tags of a title in DataFrame
medium['allTags'] = medium['title'].apply(addTags)
keep_cols = ['title', 'url', 'allTags', 'readingTime',
             'author', 'text']

# now will keep only the required columns
medium = medium[keep_cols]

# remove all rows where title is empty
trial = medium[medium['title'] != 'NaN']
medium.reset_index(drop = True, inplace = True)


def clean_text(text):
    '''
    Eliminates links, non alphanumerics, and punctuation.
    Returns lower case text.
    '''

    # Remove links
    text = re.sub('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w /\-?= %.]+','', text)
    # Remove non-alphanumerics
    text = re.sub('\w*\d\w*', ' ', text)
    # Remove
    # punctuation and lowercase
    text = re.sub('[%s]' % re.escape(string.punctuation),' ', text.lower())
    # Remove newline characters
    text = text.replace('\n', ' ')
    return text

medium['text'] = medium['text'].apply(clean_text)

# setting some data science specific stop words.
stop_list = STOPWORDS.union(set(['data', 'ai', 'learning', 'time', 'machine', 'like', 'use', 'new', 'intelligence', 'need', "it's", 'way', 'artificial', 'based', 'want', 'know', 'learn', "don't", 'things', 'lot', "let's", 'model', 'input', 'output', 'train', 'training', 'trained', 'it', 'we', 'don', 'you', 'ce', 'hasn', 'sa', 'do', 'som', 'can']))
def remove_stopwords(text):
    clean_text = []
    for word in text.split(' '):
        if word not in stop_list and (len(word) > 2):
            clean_text.append(word)
    return ' '.join(clean_text)

medium['text'] = medium['text'].apply(remove_stopwords)

# Apply stemmer to processedText
stemmer = PorterStemmer()
def stem_text(text):
    word_list = []
    for word in text.split(' '):
        word_list.append(stemmer.stem(word))
    return ' '.join(word_list)
medium['text'] = medium['text'].apply(stem_text)
#medium.to_csv('pre-processed.csv')

'''
next step is topic modelling
we will first need to convert our document into a series of word vectors
we will do that using TFIFD for nmf and svd algo
'''

vectorizer= TfidfVectorizer(stop_words=stop_list, ngram_range=(1,1))
# creating an instance of tfidvectorizer and we need to specify stopwords coz
# we are using medium['text'] here not the preprocessed document


doc_word=vectorizer.fit_transform(medium['text'])   # fit - means making calculation transform- applying calculations to data
# gives us a sparse matrix
#print(doc_word) #gives us the matrix
#print(vectorizer.get_feature_names())- gives the value of text coresspondin to the matrix
# so next step is to perform dimensionality reduction to obtain the main topics of the matrix
# algo to perform that is SVD

# applying the first topic modelling algorithm- SVD. through hit and miss decided to keep the topic number as 8
#svd=TruncatedSVD(8)  # this will extract 8 topics i.e our txt is now in the form of 8 dimensional vectors
#doc_svd= svd.fit_transform(doc_word)


def display_topics(model, feature_names, no_top_words, no_top_topics, topic_names=None):
    count = 0
    for ix, topic in enumerate(model.components_):
        if count == no_top_topics:
            break
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", (ix + 1))
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words -
                                                 1:-1]]))
        count += 1

#display_topics(svd, vectorizer.get_feature_names(), 15, 8)

# applying the second algorithm- LDA
# tokenized_docs = medium['text'].apply(simple_preprocess)
# dictionary = gensim.corpora.Dictionary(tokenized_docs)
# dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
# corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
# # Workers = 4 activates all four cores of my CPU,
# lda = models.LdaMulticore(corpus=corpus, num_topics=8,
#                           id2word=dictionary, passes=10,
#                           workers = 4)
# lda.print_topics()


# applying the 3rd algorithm- NMF
nmf = NMF(8)
docs_nmf = nmf.fit_transform(doc_word)
#display_topics(nmf, vectorizer.get_feature_names(), 15, 8)

column_names = ['title', 'url', 'allTags', 'readingTime', 'author',
                'Tech', 'Modeling', 'Bots', 'Deep Learning',
                'Coding', 'Business', 'Careers', 'NLP', 'sum']
# Create topic sum for each article
# Later remove all articles with sum 0- this is done to remove all the articles that have no relation to our topics
topic_sum = pd.DataFrame(np.sum(docs_nmf, axis = 1))
# Turn our docs_nmf array into a data frame
doc_topic_df = pd.DataFrame(data = docs_nmf)
# Merge all of our article metadata and name columns
doc_topic_df = pd.concat([medium[['title', 'url', 'allTags','readingTime', 'author']], doc_topic_df,topic_sum], axis = 1)
doc_topic_df.columns = column_names
# Remove articles with topic sum = 0, then drop sum column
doc_topic_df = doc_topic_df[doc_topic_df['sum'] != 0]  # sum will be 0 only when all topic distribution is 0, that means such articles are of no use to us. so remove it
doc_topic_df.drop(columns = 'sum', inplace = True)
# Reset index then save
doc_topic_df.reset_index(drop = True, inplace = True)
#doc_topic_df.to_csv('tfidf_nmf_8topics.csv', index = False)

topic_names = ['Tech', 'Modeling', 'Bots', 'Deep Learning','Coding', 'Business', 'Careers', 'NLP']
topic_array = np.array(doc_topic_df[topic_names])    # basically a row here contains topic distribution for a given article
#print(topic_array)
norms = np.linalg.norm(topic_array, axis = 1)

def compute_dists(top_vec, topic_array):
    '''
    Returns cosine distances for top_vec compared to every article
    '''
    dots = np.matmul(topic_array, top_vec)
    input_norm = np.linalg.norm(top_vec)
    co_dists = dots / (input_norm * norms)
    return co_dists

def produce_rec(top_vec, topic_array, doc_topic_df, rand = 15):
    '''
    Produces a recommendation based on cosine distance.
    rand controls magnitude of randomness.
    '''
    top_vec = top_vec + np.random.rand(8,)/(np.linalg.norm(top_vec)) * rand
    # here w e could simply add np.random.rand but that would lead to a lot of randomization so we divide it by the above eqn and here rand= 15 is just randomly taken , we can use
    # any other value also to minimize the randomization
    co_dists = compute_dists(top_vec, topic_array)
    # print(co_dists)
    #print(np.argmax(co_dists))
    return doc_topic_df.loc[np.argmax(co_dists)]

# for the purpose of output
def set_prefernce():
    # set the prefernce of user
    # take the input from user- left
    tech = 5
    modeling = 5
    bots = 0  #bots
    deep = 0
    coding = 0
    business = 5
    careers = 0
    nlp = 0
    top_vec = np.array([tech, modeling, bots, deep, coding, business, careers, nlp])
    return top_vec

top_vec= set_prefernce()
rec = produce_rec(top_vec, topic_array, doc_topic_df)
print(rec)