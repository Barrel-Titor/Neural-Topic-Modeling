import pandas as pd
from gensim import corpora
import gensim
pd.set_option("display.max_colwidth", 200)
from sklearn.datasets import fetch_20newsgroups
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data
l=len(documents)
news_df = pd.DataFrame({'document':documents})
# removing everything except alphabets`
news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z#]", " ")
# removing short words
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
# make all text lowercase
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
# tokenization
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())
# remove stop-words
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
# de-tokenization
dictionary = corpora.Dictionary(tokenized_doc)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
print(corpus)
# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=50, id2word=dictionary, passes=30)
print(ldamodel.print_topics(num_topics=50, num_words=8))