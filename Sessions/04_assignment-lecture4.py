# -*- coding: utf-8 -*-
"""
Solution for assignment-lecture4.ipynb

Dataset: IMDB Dataset of 50K Movie Reviews
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import re
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
# Adjust the path if needed
df = pd.read_csv('IMDB Dataset.csv')

print(f"Dataset shape: {df.shape}")
print(df.head())

# -------------------------------------------------------------------------
# Problem 1: Apply all the preprocessing techniques that are necessary
# -------------------------------------------------------------------------

def preprocess_text(text):
    """
    Apply comprehensive text preprocessing:
    1. Converting to lowercase
    2. Removing HTML tags
    3. Removing punctuation
    4. Removing stopwords
    5. Tokenizing
    6. Stemming
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    
    # Join tokens back to text for vectorization methods
    processed_text = ' '.join(stemmed_tokens)
    
    return stemmed_tokens, processed_text

# Apply preprocessing to the dataset
df['tokens'], df['processed_text'] = zip(*df['review'].apply(preprocess_text))

print("\nProblem 1: Preprocessing example")
print(f"Original text: {df['review'].iloc[0][:100]}...")
print(f"Processed text: {df['processed_text'].iloc[0][:100]}...")

# -------------------------------------------------------------------------
# Problem 2: Find out the number of words in the entire corpus and 
# also the total number of unique words(vocabulary) using just python
# -------------------------------------------------------------------------

def count_corpus_words(tokens_series):
    """
    Count total words and unique words (vocabulary size) in a series of token lists
    """
    # Flatten the list of tokens from all documents
    all_words = [word for tokens in tokens_series for word in tokens]
    
    # Count total words
    total_words = len(all_words)
    
    # Count unique words (vocabulary size)
    unique_words = len(set(all_words))
    
    return total_words, unique_words

total_words, vocab_size = count_corpus_words(df['tokens'])

print("\nProblem 2: Word counts")
print(f"Total number of words in corpus: {total_words}")
print(f"Total number of unique words (vocabulary size): {vocab_size}")

# -------------------------------------------------------------------------
# Problem 3: Apply One Hot Encoding
# -------------------------------------------------------------------------

def one_hot_encode(corpus):
    """
    Apply one-hot encoding to the corpus
    """
    # Use CountVectorizer with binary=True to get one-hot encoding
    vectorizer = CountVectorizer(binary=True)
    one_hot_vectors = vectorizer.fit_transform(corpus)
    vocab = vectorizer.get_feature_names_out()
    
    return one_hot_vectors, vocab

# Apply one-hot encoding
one_hot_vectors, one_hot_vocab = one_hot_encode(df['processed_text'])

print("\nProblem 3: One-hot encoding")
print(f"Shape of one-hot encoded matrix: {one_hot_vectors.shape}")
print(f"Example vocabulary (first 10 words): {list(one_hot_vocab)[:10]}")

# Sample of the one-hot encoded matrix for visualization
sample_matrix = one_hot_vectors[:2, :5].toarray()
print("Sample of one-hot encoded matrix (2 documents x 5 words):")
for i, doc in enumerate(sample_matrix):
    print(f"Document {i+1}: {doc}")

# -------------------------------------------------------------------------
# Problem 4: Apply bag words and find the vocabulary also find the 
# times each word has occurred
# -------------------------------------------------------------------------

def bag_of_words(corpus):
    """
    Apply bag-of-words model to the corpus and count word frequencies
    """
    # Use CountVectorizer to implement bag-of-words
    vectorizer = CountVectorizer()
    bow_vectors = vectorizer.fit_transform(corpus)
    vocab = vectorizer.get_feature_names_out()
    
    # Calculate word frequencies across the corpus
    word_counts = dict(zip(vocab, bow_vectors.sum(axis=0).A1))
    
    return bow_vectors, vocab, word_counts

# Apply bag-of-words
bow_vectors, bow_vocab, word_counts = bag_of_words(df['processed_text'])

print("\nProblem 4: Bag of Words")
print(f"Shape of bag-of-words matrix: {bow_vectors.shape}")
print(f"Vocabulary size: {len(bow_vocab)}")

# Print the most common words
most_common = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
print("10 most common words and their frequencies:")
for word, count in most_common:
    print(f"{word}: {count}")

# -------------------------------------------------------------------------
# Problem 5: Apply bag of bi-gram and bag of tri-gram and write down 
# your observation about the dimensionality of the vocabulary
# -------------------------------------------------------------------------

def ngram_analysis(corpus):
    """
    Apply n-gram analysis (unigrams, bigrams, trigrams) to the corpus
    """
    results = {}
    
    # Generate unigrams, bigrams, and trigrams
    for n in [1, 2, 3]:
        vectorizer = CountVectorizer(ngram_range=(n, n))
        vectors = vectorizer.fit_transform(corpus)
        vocab = vectorizer.get_feature_names_out()
        
        results[f"{n}-gram"] = {
            "vectors": vectors,
            "vocab": vocab,
            "vocab_size": len(vocab)
        }
    
    return results

# Apply n-gram analysis
ngram_results = ngram_analysis(df['processed_text'])

print("\nProblem 5: N-gram analysis")
# Compare vocabulary sizes
for n in [1, 2, 3]:
    print(f"Vocabulary size for {n}-gram: {ngram_results[f'{n}-gram']['vocab_size']}")
    
# Show samples of bigrams and trigrams
print("\nSample of bigrams:")
print(list(ngram_results['2-gram']['vocab'])[:10])

print("\nSample of trigrams:")
print(list(ngram_results['3-gram']['vocab'])[:10])

# Analyze dimensionality increase
unigram_size = ngram_results['1-gram']['vocab_size']
bigram_size = ngram_results['2-gram']['vocab_size']
trigram_size = ngram_results['3-gram']['vocab_size']

print("\nDimensionality analysis:")
print(f"Increase from unigram to bigram: {bigram_size / unigram_size:.2f}x")
print(f"Increase from unigram to trigram: {trigram_size / unigram_size:.2f}x")
print(f"Increase from bigram to trigram: {trigram_size / bigram_size:.2f}x")

# -------------------------------------------------------------------------
# Problem 6: Apply tf-idf and find out the idf scores of words, 
# also find out the vocabulary.
# -------------------------------------------------------------------------

def tfidf_analysis(corpus):
    """
    Apply TF-IDF vectorization to the corpus and analyze IDF scores
    """
    # Use TfidfVectorizer to implement TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_vectors = vectorizer.fit_transform(corpus)
    vocab = vectorizer.get_feature_names_out()
    
    # Extract IDF scores for each term
    idf_scores = dict(zip(vocab, vectorizer.idf_))
    
    return tfidf_vectors, vocab, idf_scores

# Apply TF-IDF
tfidf_vectors, tfidf_vocab, idf_scores = tfidf_analysis(df['processed_text'])

print("\nProblem 6: TF-IDF analysis")
print(f"Shape of TF-IDF matrix: {tfidf_vectors.shape}")
print(f"Vocabulary size: {len(tfidf_vocab)}")

# Print words with highest and lowest IDF scores
sorted_idf = sorted(idf_scores.items(), key=lambda x: x[1])

print("\nWords with lowest IDF scores (most common across documents):")
for word, score in sorted_idf[:10]:
    print(f"{word}: {score:.4f}")

print("\nWords with highest IDF scores (most unique to specific documents):")
for word, score in sorted_idf[-10:]:
    print(f"{word}: {score:.4f}")
