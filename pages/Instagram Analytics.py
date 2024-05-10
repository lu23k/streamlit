import os
import copy

import streamlit as st
#Basic libraries
import pandas as pd 
import numpy as np 

#NLTK libraries
import nltk
import re
import string
from wordcloud import WordCloud,STOPWORDS
from langdetect import detect
import unicodedata

#Visualization libraries
from textblob import TextBlob
from plotly import tools
import plotly.graph_objs as go
from plotly.subplots import make_subplots

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

#Other miscellaneous libraries
from scipy import interp
from itertools import cycle
import cufflinks as cf
from collections import defaultdict
from collections import Counter

# Use a pipeline as a high-level helper
from transformers import pipeline
model = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

def clean_text(text):
    # Make text lowercase
    text = str(text).lower()
    
    # Remove text in square brackets
    text = re.sub('\[.*?\]', '', text)
    
    # Remove links
    text = re.sub('https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub('<.*?>+', '', text)
    
    # Remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    
    # Remove newlines
    text = re.sub('\n', '', text)
    
    # Remove words containing numbers
    text = re.sub('\w*\d\w*', '', text)
    
    return text

def remove_emojis(text):
    # Split the sentence into individual characters
    characters = [char for char in text]
    
    # Iterate over each character and remove emojis
    cleaned_text = ''
    for char in characters:
        if not any(char in range(0x1F600, 0x1F650) or char in range(0x1F300, 0x1F6FF) or char in range(0x2600, 0x26FF) for char in map(ord, char)):
            cleaned_text += char
    
    return cleaned_text

def filter_english_reviews(df):
    # Function to detect language
    def detect_language(text):
        try:
            language = detect(text)
            return language == 'en'  # Return True if language is English
        except:
            return False  # Return False if language detection fails
    
    # Filter non-English comments
    df['is_english'] = df['text'].apply(detect_language)
    df = df[df['is_english']]
    
    return df

#custom function for ngram generation 
def generate_ngrams(text, n_gram=2):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

#custom function for horizontal bar chart
def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return trace

def plot_bigram(text,file_name):
    # Generate the bar chart from reviews
    freq_dict = defaultdict(int)
    for sentence in text:
        encoded_input = tokenizer(sentence, return_tensors='pt')
        # sentiment = model(sentence)
        for word in generate_ngrams(sentence,2):
            freq_dict[word] += 1


    fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    fd_sorted.columns = ["word", "wordcount"]

    # Check if the DataFrame is empty
    if fd_sorted.empty:
        st.title("No bigrams found.")
        return  # Return early if the DataFrame is empty

    trace0 = horizontal_bar_chart(fd_sorted.head(25), 'orange')

    # Create a subplot
    fig = make_subplots(rows=1, cols=1, subplot_titles=[file_name.split('_', 1)[0]])
    fig.add_trace(trace0, 1, 1)

    fig.update_layout(height=900, width=1000, title="High Frequency Words")
    st.plotly_chart(fig, use_container_width=True)



def process_reviews(raw_reviews, year, month, filename):

    # Set the index to 'Post_creation_date'
    raw_reviews.set_index('Post_creation_date', inplace=True)
    
    # Convert the index to a DatetimeIndex object
    raw_reviews.index = pd.to_datetime(raw_reviews.index)
    
    # Sort the DataFrame by index
    raw_reviews = raw_reviews.sort_index()
    
    # Clean the text
    raw_reviews['text'] = raw_reviews['text'].apply(clean_text)
    raw_reviews['text'] = raw_reviews['text'].apply(remove_emojis)
    stop_words= ['yourselves', 'between', 'whom', 'itself', 'is', "she's", 'up', 'herself', 'here', 'your', 'each', 
             'we', 'he', 'my', "you've", 'having', 'in', 'both', 'for', 'themselves', 'are', 'them', 'other',
             'and', 'an', 'during', 'their', 'can', 'yourself', 'she', 'until', 'so', 'these', 'ours', 'above', 
             'what', 'while', 'have', 're', 'more', 'only', "needn't", 'when', 'just', 'that', 'were', "don't", 
             'very', 'should', 'any', 'y', 'isn', 'who',  'a', 'they', 'to', 'too', "should've", 'has', 'before',
             'into', 'yours', "it's", 'do', 'against', 'on',  'now', 'her', 've', 'd', 'by', 'am', 'from', 
             'about', 'further', "that'll", "you'd", 'you', 'as', 'how', 'been', 'the', 'or', 'doing', 'such',
             'his', 'himself', 'ourselves',  'was', 'through', 'out', 'below', 'own', 'myself', 'theirs', 
             'me', 'why', 'once',  'him', 'than', 'be', 'most', "you'll", 'same', 'some', 'with', 'few', 'it',
             'at', 'after', 'its', 'which', 'there','our', 'this', 'hers', 'being', 'did', 'of', 'had', 'under',
             'over','again', 'where', 'those', 'then', "you're", 'i', 'because', 'does', 'all','link','join','bio','us','we',
            'myself','we','you','de','la','link','bio']
    raw_reviews['text'] = raw_reviews['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    
    # Filter English reviews
    processed_reviews = filter_english_reviews(raw_reviews)
    
    # Filter the reviews based on the specified year and quarter
    filtered_reviews = raw_reviews[(raw_reviews.index.year == year) & (raw_reviews.index.month == month)]
    
    if filtered_reviews.empty:
        st.title("No available data")
        return  # Return early if the DataFrame is empty

    #plot bigram
    plot_bigram(filtered_reviews['text'],filename)

#set header
st.header(":rainbow[Instagram Hashtag Analytics] ", divider='rainbow')
#import file
file = st.sidebar.file_uploader("Import File")
year = st.sidebar.selectbox('Year', options=['2023', '2024'], index=1)
month = st.sidebar.selectbox('Month', options=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'], index=0)

if file:

    #get data 
    @st.cache_data
    def load_data(path):
        df = pd.read_csv(path)
        # df.columns = df.columns.str.lower()
        return df

    df = load_data(file)
    process_reviews(df, int(year), int(month),file.name)


else:

    st.write("Please upload your hashtag file")


