# world cloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import numpy as np
import os
from PyPDF2 import PdfReader

# read pdf
pdf_path  = '../OMERS-2023-Annual-Report.pdf'
reader = PdfReader(pdf_path)
full_text = ''
for page in reader.pages:
    full_text += page.extract_text() + ' '
# generate word cloud and filter out common words
# Preprocess the text for keyword extraction: convert to lowercase, remove punctuation, etc.
import re
from collections import Counter

# Remove numbers and punctuation
text_processed = re.sub('[^a-zA-Z\s]', '', full_text.lower())
# Tokenize the text into words
words = text_processed.split()

# Count the occurrences of each word
word_counts = Counter(words)

# List of common English stopwords
stopwords = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
             "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
             "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
             "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
             "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
             "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
             "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
             "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
             "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
             "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve",
             "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn",
             "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn"
             , "annual"}
# Filter out the stopwords from our word counts
filtered_word_counts = {word: count for word, count in word_counts.items() if word not in stopwords}

# Generate a word cloud image
wordcloud = WordCloud(width=800, height=400, background_color ='white', stopwords=STOPWORDS
                      ).generate_from_frequencies(filtered_word_counts)

# Display the generated image:
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
# save the word cloud
wordcloud.to_file("wordcloud.png")

# correlation
