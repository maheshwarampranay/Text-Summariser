# summarizer.py

import nltk
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import numpy as np
import pdfplumber
from docx import Document

nltk.download('punkt')

# Load pretrained abstractive summarizer
abstractive_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def preprocess(text):
    return text.replace("\n", " ").strip()

def abstractive_summary(text, max_length=150, min_length=40):
    return abstractive_summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

def extractive_summary(text, num_sentences=3):
    sentences = sent_tokenize(text)
    tfidf = TfidfVectorizer().fit_transform(sentences)
    scores = np.array(tfidf.sum(axis=1)).flatten()
    ranked_sentences = [sentences[i] for i in scores.argsort()[-num_sentences:][::-1]]
    return ' '.join(ranked_sentences)

def read_file(file):
    if file.name.endswith('.txt'):
        return file.read().decode("utf-8")
    elif file.name.endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            return '\n'.join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif file.name.endswith('.docx'):
        doc = Document(file)
        return '\n'.join([para.text for para in doc.paragraphs])
    else:
        return "Unsupported file type."
