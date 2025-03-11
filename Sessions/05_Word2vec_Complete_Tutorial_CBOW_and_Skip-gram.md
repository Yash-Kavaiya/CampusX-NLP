# Word2vec Complete Tutorial | CBOW and Skip-gram

# Word Embeddings in NLP: A Comprehensive Guide 🔤➡️🔢

## Introduction 📚

Word embeddings are vector representations of words that capture semantic and syntactic meaning in a continuous vector space. Unlike traditional one-hot encodings, these dense vectors place words with similar meanings closer together in the high-dimensional space, effectively translating the complex relationships between words into geometric relationships.

> 💡 **Key Insight**: Word embeddings transform discrete symbolic representations (words) into continuous vector spaces where meaningful semantic operations become possible.

## Why Word Embeddings? 🤔

Traditional text representation methods have significant limitations:

| Method | Representation | Limitations |
|--------|----------------|-------------|
| One-Hot Encoding | [0,0,1,0,...,0] | ❌ High dimensionality<br>❌ No semantic information<br>❌ Sparse vectors |
| Bag of Words | Count vectors | ❌ Ignores word order<br>❌ No semantic similarity |
| TF-IDF | Weighted counts | ❌ Still misses semantic relationships |
| Word Embeddings | Dense vectors | ✅ Captures semantics<br>✅ Lower dimensionality<br>✅ Enables arithmetic operations |

## Key Properties of Word Embeddings 🌟

1. **Dimensionality Reduction**: Condenses vocabulary (potentially millions of words) into vectors of typically 50-300 dimensions

2. **Semantic Relationships**: Captures meaningful relationships between words
   ```
   king - man + woman ≈ queen
   ```

3. **Contextual Similarity**: Words used in similar contexts have similar embeddings
   ```
   sim(coffee, tea) > sim(coffee, automobile)
   ```

4. **Analogical Reasoning**: Enables solving word analogies through vector arithmetic
   ```
   vector('Paris') - vector('France') + vector('Italy') ≈ vector('Rome')
   ```

---

## Types of Word Embeddings 🔄

Word embeddings broadly fall into two categories:

### 1. Count-Based Embeddings 🧮

Count-based methods rely on **document statistics** and typically involve these steps:

1. Build a word-context co-occurrence matrix
2. Apply dimensionality reduction techniques
3. Generate dense vector representations

#### Key Count-Based Methods:

- **Latent Semantic Analysis (LSA)** / **Latent Semantic Indexing (LSI)**
  ```python
  from sklearn.decomposition import TruncatedSVD
  from sklearn.feature_extraction.text import CountVectorizer
  
  # Create co-occurrence matrix
  vectorizer = CountVectorizer()
  X = vectorizer.fit_transform(documents)
  
  # Apply SVD for dimensionality reduction
  svd = TruncatedSVD(n_components=100)
  word_embeddings = svd.fit_transform(X)
  ```

- **GloVe (Global Vectors)**
  - Combines count-based and prediction-based approaches
  - Uses a weighted least squares model to train on global word-word co-occurrence statistics
  - Outperforms pure count-based methods by capturing global statistics

#### Characteristics of Count-Based Embeddings:

| Advantages | Disadvantages |
|------------|---------------|
| ✅ Efficiently leverage statistical information | ❌ Often requires large matrices |
| ✅ Good with global co-occurrence statistics | ❌ Primarily based on document-level co-occurrence |
| ✅ Often more interpretable | ❌ Can struggle with rare words |

### 2. Prediction-Based Embeddings 🔮

Prediction-based methods use **neural networks** to predict words based on their contexts:

#### Key Prediction-Based Methods:

- **Word2Vec** (by Google)
  - Two architectures:
    - **CBOW (Continuous Bag of Words)**: Predicts target word from context words
    - **Skip-gram**: Predicts context words from target word
  
  ```python
  from gensim.models import Word2Vec
  
  # Train skip-gram model
  model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, sg=1)
  
  # Get vector for specific word
  vector = model.wv['computer']
  ```

- **FastText** (by Facebook)
  - Extension of Word2Vec that treats each word as composed of character n-grams
  - Can generate embeddings for out-of-vocabulary words
  - Better for morphologically rich languages

- **ELMo, BERT, GPT** etc.
  - Contextual embeddings (different from traditional static embeddings)
  - Generate different vectors for the same word based on its context
  - Based on deep bidirectional language models

#### Characteristics of Prediction-Based Embeddings:

| Advantages | Disadvantages |
|------------|---------------|
| ✅ Capture local context effectively | ❌ Can be computationally intensive |
| ✅ Often perform better in downstream tasks | ❌ Require significant training data |
| ✅ Better with analogical reasoning | ❌ Less interpretable than count-based methods |

---

## Comparison: Count-Based vs. Prediction-Based ⚖️

| Aspect | Count-Based | Prediction-Based |
|--------|-------------|------------------|
| Core Approach | Matrix factorization | Neural network training |
| Training Objective | Reconstruct co-occurrence statistics | Predict words from context |
| Computation | One-time matrix operation | Iterative training |
| Context Handling | Global statistics | Local windows |
| Memory Requirements | Higher (sparse matrices) | Lower (mini-batch training) |
| Scalability | Challenging with large vocabularies | More scalable |
| Representative Models | LSA, GloVe | Word2Vec, FastText |

## Mathematical Foundations 🧮

### Count-Based Approach (LSA Example):

1. Create term-document matrix X where X_ij = frequency of term i in document j
2. Apply SVD: X = USV^T
3. Reduce dimensions by keeping only k largest singular values
4. Word embeddings = U[:, :k]

### Prediction-Based Approach (Word2Vec Example):

For Skip-gram, the objective is to maximize:

```
J(θ) = 1/T ∑ᵗ ∑ⱼ log p(wₜ₊ⱼ|wₜ)
```

Where p(wₜ₊ⱼ|wₜ) is modeled as:

```
p(wₒ|wᵢ) = exp(v'ᵂᵒ · vᵂᶦ) / ∑ᵂ exp(v'ᵂ · vᵂᶦ)
```

## Evaluation Methods 📊

Word embeddings can be evaluated through:

1. **Intrinsic Evaluation**
   - Word similarity tasks (WordSim353, SimLex-999)
   - Word analogy tasks (man:woman :: king:?)
   - Categorization tests

2. **Extrinsic Evaluation**
   - Performance on downstream NLP tasks:
     - Named Entity Recognition
     - Sentiment Analysis
     - Machine Translation
     - Question Answering

## Applications of Word Embeddings 🚀

Word embeddings have revolutionized many NLP tasks:

- **Semantic Search**: Finding documents based on meaning rather than exact keyword matching
- **Machine Translation**: Improved cross-lingual mappings
- **Sentiment Analysis**: Better understanding of sentiment-bearing phrases
- **Document Classification**: Enhanced feature representation for classifiers
- **Text Summarization**: Better semantic understanding of content
- **Recommendation Systems**: Content-based filtering using semantic similarity

## Limitations and Challenges 🚧

1. **Polysemy**: Basic embeddings assign a single vector to words with multiple meanings
   - Example: "bank" (financial institution vs. river bank)

2. **Bias**: Embeddings can inherit social biases present in training data
   - Example: Gender or racial biases in analogies

3. **Contextual Understanding**: Traditional embeddings don't account for context-dependent meanings
   - Modern approaches like BERT address this with contextual embeddings

4. **Out-of-Vocabulary Words**: Classic models struggle with unseen words
   - Approaches like FastText (subword embeddings) help mitigate this

## Recent Advances: Contextual Embeddings 🔄

Modern NLP has moved toward contextual embeddings:

```
Traditional: word → vector
Contextual: word + context → vector
```

Examples include:
- **ELMo**: Bidirectional LSTM with character convolutions
- **BERT**: Bidirectional Transformer models
- **GPT**: Autoregressive Transformer models

These models generate different embeddings for the same word based on its context in the sentence.

---

## Summary 📝

Word embeddings transform discrete words into continuous vector spaces where semantic relationships are preserved as geometric relationships. The two primary approaches are:

1. **Count-Based**: Utilizing statistical co-occurrence information (LSA, GloVe)
2. **Prediction-Based**: Using neural networks to predict contextual words (Word2Vec, FastText)

# Word2Vec: Vector Representation of Words 🔤➡️🔢

## What is Word2Vec? 🤔

**Word2Vec** is a groundbreaking technique for learning word embeddings using shallow neural networks, introduced by Tomas Mikolov and his team at Google in 2013. 

> 💡 **Core Concept**: Word2Vec transforms words into numerical vector representations where semantically similar words are positioned closer together in vector space.

### How Word2Vec Works ⚙️

Word2Vec employs two primary neural network architectures:

1. **CBOW (Continuous Bag of Words)** 📦
   - Predicts target word from surrounding context words
   - Input: Context words → Output: Target word
   - Generally better for frequent words

2. **Skip-gram** ⏭️
   - Predicts context words given a target word
   - Input: Target word → Output: Context words
   - Better for rare words and small datasets

![Word2Vec Architectures](https://i.imgur.com/yWFPkP7.png)

### Training Process 🏋️‍♂️

1. Initialize random word vectors
2. Train neural network on text corpus (predict words from context or vice versa)
3. Discard the neural network, keeping only the learned word vectors
4. Result: Dense vector representations that encode semantic relationships

---

## Benefits of Word2Vec 🌟

| Benefit | Description | Impact |
|---------|-------------|--------|
| **Semantic Meaning** | Captures relationships between words | Words with similar meanings have similar vectors |
| **Dimensionality Reduction** | Represents vocabulary in dense vectors (typically 100-300 dimensions) | Efficient computation and storage |
| **Vector Arithmetic** | Enables mathematical operations on words | Allows for analogy solving and semantic reasoning |
| **Transferability** | Pre-trained vectors can be used across applications | Reduces training time for new NLP tasks |
| **Performance Boost** | Improves results on various NLP tasks | Enhances classification, clustering, and more |

---

## Real-World Applications 🚀

- **Search Improvement**: Better matching of queries with relevant documents
- **Recommendation Systems**: Finding related content based on semantic similarity
- **Document Classification**: Enhanced feature representation
- **Sentiment Analysis**: Better understanding of sentiment expressions
- **Machine Translation**: Improved cross-lingual mappings

---

## Word2Vec Demo Explained 🧪

Let's break down the code example you provided:

```python
# Importing necessary libraries
import gensim
from gensim.models import Word2Vec, KeyedVectors

# Installing and downloading pre-trained model
!pip install wget
!wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"

# Loading pre-trained model (limiting to 500,000 words for memory efficiency)
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', 
                                          binary=True, limit=500000)
```

### The Google News Model 📰

```
We will use the pre-trained weights of word2vec that was trained on Google News corpus 
containing 3 billion words. This model consists of 300-dimensional vectors 
for 3 million words and phrases.
```

- **Corpus**: Google News (3 billion words)
- **Vector Size**: 300 dimensions
- **Vocabulary**: 3 million words and phrases
- **Training Method**: Negative sampling (indicated by "negative300" in filename)

### Key Operations Demonstrated 🔍

#### 1. Finding Similar Words

```python
model.most_similar('man')
model.most_similar('cricket')
```

This returns words most similar to "man" and "cricket" based on cosine similarity:

**Example Output for 'man':**
```
[('woman', 0.768), ('boy', 0.687), ('teenager', 0.650), ...]
```

**Example Output for 'cricket':**
```
[('baseball', 0.675), ('football', 0.654), ('rugby', 0.621), ...]
```

#### 2. Computing Word Similarity

```python
model.similarity('man', 'woman')
```

Returns the cosine similarity between two word vectors (value between -1 and 1):
- Values closer to 1 indicate high similarity
- Example output: `0.768` (indicating strong semantic relationship)

#### 3. Vector Arithmetic for Analogies ✨

```python
# King - Man + Woman = ?
vec = model['king'] - model['man'] + model['woman']
model.most_similar([vec])
```

**Example Output:**
```
[('queen', 0.756), ('princess', 0.674), ('monarch', 0.571), ...]
```

This demonstrates Word2Vec's ability to solve analogies through vector arithmetic:
- `king` is to `man` as `queen` is to `woman`

```python
# Currency Analogy: INR - India + England = ?
vec = model['INR'] - model['India'] + model['England']
model.most_similar([vec])
```

**Expected Output:**
```
[('GBP', 0.723), ('pound_sterling', 0.701), ('British_pound', 0.691), ...]
```

This shows how Word2Vec captures country-currency relationships:
- INR is to India as GBP (British Pound) is to England

---

## Vector Space Visualization 📊

Word2Vec places semantically similar words near each other in vector space:

```
                Currency Space
                     $
                     |
                    USD
                   /   \
              EUR /     \ GBP
                 /       \
            INR •         • JPY
               /           \
           India           Japan
                \         /
                 \       /
                  \     /
                   \   /
                  Countries
```

---

## Implementing Word2Vec in Practice 💻

### Training a Custom Model

```python
from gensim.models import Word2Vec

# Tokenized sentences
sentences = [['I', 'love', 'natural', 'language', 'processing'], 
             ['Word', 'embeddings', 'are', 'powerful'], ...]

# Training a model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)

# Save model
model.save("word2vec.model")
```

### Using Pre-trained Models for Transfer Learning

```python
# Load pre-trained vectors
word_vectors = KeyedVectors.load_word2vec_format('pretrained_vectors.bin', binary=True)

# Initialize model with pre-trained vectors
model = Word2Vec(vector_size=300)
model.build_vocab(sentences)
model.wv.vectors = word_vectors  # Use pre-trained vectors

# Continue training on new data
model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
```

---

## Limitations and Considerations ⚠️

1. **Static Representations**: Each word has only one vector regardless of context
   - Cannot handle polysemy (e.g., "bank" as financial institution vs. river bank)
   
2. **Out-of-Vocabulary Words**: Cannot represent words not seen during training
   - Workaround: Use character-level models like FastText

3. **Training Requirements**: Needs large corpus for quality embeddings
   - Pre-trained vectors recommended for most applications

4. **Bias**: May inherit social biases present in training data
   - Requires careful evaluation and possibly debiasing techniques

---

## Evolution of Word Embeddings Timeline 📈

| Year | Model | Innovation |
|------|-------|------------|
| 2013 | Word2Vec | Efficient neural word embeddings |
| 2014 | GloVe | Global word-word co-occurrence statistics |
| 2016 | FastText | Subword embeddings for OOV words |
| 2018 | ELMo | Context-dependent embeddings |
| 2018-2019 | BERT, GPT | Transformer-based contextual embeddings |
| 2020+ | GPT-3, etc. | Large language models with deeper semantic understanding |

---

## Summary: Why Word2Vec Revolutionized NLP 🚀

Word2Vec transformed NLP by:

1. Creating dense, meaningful word representations
2. Enabling semantic operations on words through vector arithmetic
3. Improving performance across various NLP tasks
4. Providing an efficient way to transfer linguistic knowledge
5. Paving the way for modern contextual embeddings and language models
# Word2Vec Intuition: Vector Operations 🧠➡️🔢

## Vector Representation of Words 📊

The image shows a fascinating **conceptual demonstration** of how Word2Vec represents words as vectors and enables mathematical operations between them. Dated December 22, 2021, this handwritten example illustrates one of the most powerful aspects of word embeddings.

> 💡 **Key Insight**: Words can be represented as vectors in a multi-dimensional space where semantic relationships become mathematical operations.

## Semantic Feature Matrix 📋

| # | Word | Gender | Royalty | Weight | Speech |
|:-:|:----:|:------:|:-------:|:------:|:------:|
| 1 | King | 1 | 1 | 1 | 0.8 |
| 2 | Queen | 1 | 1 | 0.2 | 0.8 |
| 3 | Man | 0 | 0.3 | 0.7 | 1 |
| 4 | Woman | 0 | 0.3 | 0.2 | 1 |
| 5 | Monkey | 0 | 0 | 0.5 | 0 |

## Vector Arithmetic Demonstration ✨

The right side of the image shows the famous vector operation:

**King - Man + Woman = ?**

```
King   = [1, 1, 1, 0.8]
Man    = [0, 0.3, 0.7, 1]
Woman  = [0, 0.3, 0.2, 1]
```

### Calculation Breakdown ➗

| Dimension | Operation | Result |
|:---------:|:---------:|:------:|
| Gender | 1 - 0 + 0 | 1 |
| Royalty | 1 - 0.3 + 0.3 | 1 |
| Weight | 1 - 0.7 + 0.2 | 0.5 |
| Speech | 0.8 - 1 + 1 | 0.8 |

### Result Vector 🎯

```
Result = [1, 1, 0.5, 0.8]
```

This result vector is closest to **Queen** [1, 1, 0.2, 0.8], demonstrating how Word2Vec captures semantic relationships!

## What This Demonstrates 🔍

This example illustrates the fundamental principle behind Word2Vec's power:

1. **Semantic Preservation**: The gender and royalty dimensions are perfectly preserved
2. **Feature Transfer**: The female weight characteristic is transferred (replacing male weight)
3. **Analogy Solving**: "King is to Man as Queen is to Woman" becomes a mathematical equation

## Practical Implications 🚀

- Vector representations enable **mathematical reasoning with words**
- Semantic relationships become **geometric relationships** in vector space
- Complex linguistic concepts can be modeled through **simple vector operations**
- These properties make Word2Vec extremely valuable for applications like:
  - 🔎 Semantic search
  - 🔄 Machine translation
  - 📊 Text classification
  - 🧩 Question answering

# The Distributional Hypothesis in Word2Vec 🔤🧠

> 💡 "You shall know a word by the company it keeps." — J.R. Firth, 1957

## Core Principle: The Distributional Hypothesis 🌟

You've highlighted the fundamental principle behind Word2Vec's effectiveness:

<blockquote>
The underlying assumption of Word2Vec is that two words sharing similar contexts also share a similar meaning and consequently a similar vector representation from the model.
</blockquote>

This principle, formally known as the **Distributional Hypothesis**, serves as the theoretical foundation for most modern word embedding techniques.

## How Word2Vec Implements This Principle 🛠️

| Model Architecture | Training Objective | Context Definition |
|:------------------:|:------------------:|:------------------:|
| **CBOW** | Predict target word from surrounding context | Fixed window of words around target |
| **Skip-gram** | Predict context words from target word | Fixed window of words around target |

Both architectures optimize word vectors so that words appearing in similar contexts will have similar vector representations.

## Visualizing the Context Window 🔍

```
Sentence: "The quick brown fox jumps over the lazy dog"
                   ↑
                 Target
                   ↓
Context Window (size=2): ["quick", "brown", "jumps", "over"]
```

During training, Word2Vec adjusts vector representations so:
- Words that frequently appear in this context window will have similar vectors
- Words rarely appearing in similar contexts will have dissimilar vectors

## Evidence in Vector Space 📊

The distributional hypothesis manifests as observable patterns in vector space:

| Relationship Type | Vector Space Pattern | Example |
|-------------------|----------------------|---------|
| **Synonyms** | Near-identical vectors | good ≈ great |
| **Antonyms** | Similar contexts, different meanings | hot ↔️ cold |
| **Analogies** | Consistent vector differences | king - man + woman ≈ queen |
| **Semantic Categories** | Cluster formation | apple, orange, banana form fruit cluster |

## Mathematical Expression 🧮

For words $w_i$ and $w_j$ with context sets $C(w_i)$ and $C(w_j)$:

$$similarity(w_i, w_j) \propto |C(w_i) \cap C(w_j)|$$

Word2Vec approximates this through vector similarity:

$$similarity(w_i, w_j) \approx \frac{\vec{v_i} \cdot \vec{v_j}}{|\vec{v_i}||\vec{v_j}|}$$

## Practical Implications 🚀

The distributional hypothesis enables Word2Vec to:

- 🔄 **Transfer Knowledge**: Words not seen together during training can still be related
- 🔍 **Discover Relations**: Automatically detect semantic relationships
- 📈 **Generalize**: Apply learned relationships to new words and contexts
- 🌐 **Cross-Domain Application**: Work across different text domains and languages

## Limitations & Considerations ⚠️

| Limitation | Description | Potential Solution |
|------------|-------------|-------------------|
| **Polysemy** | Single vector for words with multiple meanings | Contextual embeddings (BERT, ELMo) |
| **Rare Words** | Poor vectors for infrequent words | Subword embeddings (FastText) |
| **Domain Specificity** | Different meanings in different domains | Domain-specific training |
| **Window Size** | Fixed context may miss long-range dependencies | Transformer models |

## Evolution of the Distributional Approach 📈

```mermaid
graph LR
    A[Count-Based] --> B[Word2Vec]
    B --> C[GloVe] 
    C --> D[FastText]
    D --> E[Contextual Embeddings]
    E --> F[Large Language Models]
```

## Testing the Hypothesis 🔬

You can validate the distributional hypothesis by:

1. 📝 **Corpus Analysis**: Measure co-occurrence patterns of semantically related words
2. 🔢 **Vector Similarity**: Calculate cosine similarity between semantically related words
3. 🧩 **Analogy Tasks**: Test vector arithmetic on semantic relationships
4. 🎯 **Downstream Tasks**: Evaluate performance in applications like classification
