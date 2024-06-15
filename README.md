# BERT Fine-tuned for Named Entity Recognition
This repository contains code for fine-tuning the BERT (Bidirectional Encoder Representations from Transformers) model on Named Entity Recognition (NER) tasks using the Hugging Face Transformers library.

## Overview
Named Entity Recognition (NER) is a fundamental task in natural language processing (NLP) that involves identifying and classifying named entities (such as persons, organizations, and locations) within text. This project utilizes BERT, a powerful pre-trained language model, fine-tuned on the CoNLL 2003 dataset to achieve state-of-the-art performance in NER classification.

## Why BERT?
The notebook briefly showcases word embeddings that are from 'word2vec-google-news-300', which is a pre-trained word embedding model developed using the word2vec algorithm on a dataset derived from Google News articles. These embeddings are fixed, context-independent representations of whole words, where each word is mapped to a continuous vector in a 300-dimensional space.
In contrast, to perform Named Entity Recognition (NER) classification, the notebook fine-tunes BERT (Bidirectional Encoder Representations from Transformers). BERT uses dynamic, context-sensitive embeddings known as WordPiece embeddings.
Besides, BERT utilizes a transformer architecture that processes the entire input sequence bidirectionally. This allows BERT to consider both left and right context for each token, capturing deeper semantic relationships and contextual nuances within the text.
This approach enhances BERT's ability to understand and process a wide range of texts with varying linguistic complexities, making it highly effective for tasks requiring deep contextual understanding, such as NER classification."
## Usage

### Inference

To run inference using this model, load the model with Transformers pipeline for NER.
```
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("jh-hoo/bert-finetuned-ner")
model = AutoModelForTokenClassification.from_pretrained("jh-hoo/bert-finetuned-ner")

class_ner = pipeline("ner", model=model, tokenizer=tokenizer)
example = "The headquarters of Google is located in Mountain View, California."

ner_results = class_ner(example)
print(ner_results)
```
