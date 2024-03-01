from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt
import re
from collections import defaultdict
import os
from scipy.sparse import csr_matrix
import numpy as np
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from gensim.models import TfidfModel
import pyLDAvis.gensim_models as gensimvis

#Preprocessing and Bag of Words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
documents = []
folder_path = 'csabstracts-master/'

ps = PorterStemmer()

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            documents.extend(file.readlines())

def tokenize_text(text):
    text = ps.stem(text)
    text = text.split(None,2)
    return text[1],re.findall(r'\b\w+\b',text[2].lower())

label_list = []
tokenized_list = []
for doc in documents:
    label,tokenized_document = tokenize_text(doc)
    label_list.append(label)
    tokenized_list.append(tokenized_document)

vocabulary = set(word for tokens in tokenized_list for word in tokens)
word_to_id = {word: id for id, word in enumerate(vocabulary)}
id_to_word = {id: word for word, id in word_to_id.items()}

document_vectors = []
for tokens in tokenized_list:
    vector = defaultdict(int)
    for token in tokens:
        vector[word_to_id[token]] += 1
    document_vectors.append(vector)

num_of_docs = len(documents)
num_of_words = len(vocabulary)
data,rows,columns = [], [], []
for i, vector in enumerate(document_vectors):
    for word_id, count in vector.items():
        rows.append(i)
        columns.append(word_id)
        data.append(count)
        #print(f"{id_to_word[word_id]}: {count}")
print('---------------------------------------------------------------------------------------------------------------------')
bag_of_words_matrix = csr_matrix((data, (rows, columns)), shape=(num_of_docs, num_of_words))


print("Everything from here on out is generated from ChatGPT. I'm sorry :(")
#Naive Bayes
categories = label_list  # This should come from your document processing

tfidf_transformer = TfidfTransformer()

# Fit and transform the bag-of-words matrix to get TF-IDF
tfidf_matrix = tfidf_transformer.fit_transform(bag_of_words_matrix)

# Proceed with your analysis using 'tfidf_matrix' instead of 'bag_of_words_matrix'
# For example, converting tfidf_matrix to a lil_matrix for easier manipulation
tfidf_lil = tfidf_matrix.tolil()

# Update counts for each category and word-category combinations using TF-IDF values
category_counts_tfidf = defaultdict(int)
word_category_counts_tfidf = defaultdict(lambda: defaultdict(int))

for doc_idx, category in enumerate(categories):
    for word_id in tfidf_lil.rows[doc_idx]:
        word_category_counts_tfidf[category][word_id] += tfidf_lil[doc_idx, word_id]
        category_counts_tfidf[category] += tfidf_lil[doc_idx, word_id]

# Convert bag_of_words_matrix to a format that's easier to work with for summing
bow_matrix = bag_of_words_matrix.tolil()

# Initialize count structures
category_counts = defaultdict(int)
word_category_counts = defaultdict(lambda: defaultdict(int))
total_word_counts = defaultdict(int)

# Calculate counts for each category and word-category combinations
for doc_idx, category in enumerate(categories):
    for word_id in bow_matrix.rows[doc_idx]:
        word_category_counts[category][word_id] += bow_matrix[doc_idx, word_id]
        category_counts[category] += bow_matrix[doc_idx, word_id]
        total_word_counts[word_id] += bow_matrix[doc_idx, word_id]

# Initialize structures to store probabilities
P_w_c = defaultdict(lambda: defaultdict(float))
P_w_Co = defaultdict(float)

# Calculate P(w|c) for each word in each category
for category, word_counts in word_category_counts.items():
    total_count = sum(word_counts.values())
    for word_id, count in word_counts.items():
        P_w_c[category][word_id] = (count + 1) / (total_count + len(vocabulary))

# Calculate P(w|C_o) for each word across all other categories
for word_id in total_word_counts:
    P_w_Co[word_id] = sum((word_category_counts[other_category][word_id] for other_category in categories if other_category != category), 1) / (sum((category_counts[other_category] for other_category in categories if other_category != category), len(vocabulary)))
    
    '''
    P_w_Co[word_id] = 
    sum((word_category_counts[other_category][word_id] for other_category in categories if other_category != category), 1) / 
    (sum((category_counts[other_category] for other_category in categories if other_category != category), len(vocabulary)))
    '''

# Compute log likelihood ratios
log_likelihood_ratios = defaultdict(lambda: defaultdict(float))
for category, word_probs in P_w_c.items():
    for word_id, prob in word_probs.items():
        P_w_co = P_w_Co[word_id]
        log_likelihood_ratios[category][word_id] = np.log(prob / P_w_co)

# Identify top 10 words for each category
top_words_per_category = {category: sorted(word_probs.keys(), key=lambda word_id: log_likelihood_ratios[category][word_id], reverse=True)[:10] for category, word_probs in P_w_c.items()}

'''
top_words_per_category = 
{category: sorted(word_probs.keys(), key=lambda word_id: log_likelihood_ratios[category][word_id], reverse=True)[:10] 
for category, word_probs in P_w_c.items()}
'''

# Display top words for each category (convert word IDs back to words correctly)
for category, top_word_ids in top_words_per_category.items():
    top_words = [id_to_word[word_id] for word_id in top_word_ids]
    print(f"Category: {category}, Top words: {top_words}")
print('---------------------------------------------------------------------------------------------------------------------')

dictionary = Dictionary(tokenized_list)
corpus = [dictionary.doc2bow(text) for text in tokenized_list]

num_topics = 10
lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

'''
lda = 
LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100, 
update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
'''

topics = lda.print_topics(num_words=10)
for topic in topics:
    print(topic)
print('---------------------------------------------------------------------------------------------------------------------')

'''
Analyze Topic Distribution Across Categories
For each category, compute the average topic distribution:
'''
category_topic_distribution = defaultdict(lambda: np.zeros(num_topics))

for i, row in enumerate(corpus):
    category = categories[i]
    for topic_num, proportion_topic in lda.get_document_topics(row):
        category_topic_distribution[category][topic_num] += proportion_topic

# Average the topic distribution
for category in category_topic_distribution:
    category_topic_distribution[category] /= category_counts[category]

# Print top topics for each category
for category, distribution in category_topic_distribution.items():
    top_topics = sorted(range(num_topics), key=lambda i: distribution[i], reverse=True)[:5]
    print(f"Category: {category}, Top Topics: {top_topics}")
print('---------------------------------------------------------------------------------------------------------------------')

print('Computing for Tf-idf next')
# Calculate P(w|c) for each word in each category using TF-IDF
for category, word_id_tfidf_sum in word_category_counts_tfidf.items():
    total_tfidf_sum = sum(word_id_tfidf_sum.values())
    for word_id, tfidf_value in word_id_tfidf_sum.items():
        P_w_c[category][word_id] = (tfidf_value + 1) / (total_tfidf_sum + len(vocabulary))

# Calculate P(w|C_o) for each word across all other categories using TF-IDF
for word_id in total_word_counts.keys():
    P_w_Co[word_id] = sum((word_category_counts_tfidf[other_category][word_id] for other_category in categories if other_category != category), 1) / (sum((category_counts_tfidf[other_category] for other_category in categories if other_category != category), len(vocabulary)))

# Compute log likelihood ratios using TF-IDF values
for category, word_probs in P_w_c.items():
    for word_id, prob in word_probs.items():
        P_w_co = P_w_Co[word_id]
        # Guard against division by zero in case P_w_co is 0
        log_likelihood_ratios[category][word_id] = np.log(prob / P_w_co) if P_w_co > 0 else 0

# Identify top 10 words for each category using log likelihood ratios based on TF-IDF
top_words_per_category_tfidf = {category: sorted(word_probs.keys(), key=lambda word_id: log_likelihood_ratios[category][word_id], reverse=True)[:10] for category, word_probs in P_w_c.items()}

# Display top words for each category (convert word IDs back to words) using TF-IDF
for category, top_word_ids in top_words_per_category_tfidf.items():
    top_words_tfidf = [id_to_word[word_id] for word_id in top_word_ids]
    print(f"Category: {category}, Top words (TF-IDF): {top_words_tfidf}")
print('---------------------------------------------------------------------------------------------------------------------')

# Assuming 'dictionary' and 'corpus' (bag-of-words) are already defined
# Create a TF-IDF model and transform the corpus
tfidf = TfidfModel(corpus)  # Train a TfidfModel on the corpus
tfidf_corpus = tfidf[corpus]  # Transform the corpus to TF-IDF weights

# Train the LDA model using the TF-IDF corpus instead of the raw frequencies
lda_tfidf = LdaModel(corpus=tfidf_corpus, id2word=dictionary, num_topics=num_topics, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

# Print topics with TF-IDF weights
topics_tfidf = lda_tfidf.print_topics(num_words=10)
for topic in topics_tfidf:
    print(topic)
print('---------------------------------------------------------------------------------------------------------------------')

# Adapt the topic distribution analysis to use the TF-IDF transformed corpus
category_topic_distribution_tfidf = defaultdict(lambda: np.zeros(num_topics))

for i, row in enumerate(tfidf_corpus):  # Iterate over the TF-IDF corpus
    category = categories[i]
    for topic_num, proportion_topic in lda_tfidf.get_document_topics(row):
        category_topic_distribution_tfidf[category][topic_num] += proportion_topic

# Average the topic distribution for TF-IDF
for category in category_topic_distribution_tfidf:
    category_topic_distribution_tfidf[category] /= category_counts[category]

# Print top topics for each category using TF-IDF
for category, distribution in category_topic_distribution_tfidf.items():
    top_topics_tfidf = sorted(range(num_topics), key=lambda i: distribution[i], reverse=True)[:5]
    print(f"Category: {category}, Top Topics (TF-IDF): {top_topics_tfidf}")
