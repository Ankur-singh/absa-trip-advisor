import re
import nltk
from nltk import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from nrclex import NRCLex
from textblob import TextBlob

import spacy_streamlit
import spacy

st.set_page_config(page_title="Review Analysis",
                    page_icon="👩‍💻",
                    layout="wide",
                    initial_sidebar_state="expanded")

nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def process_review(review):
    # if expression in the sentence is not a word then this code change them to space
    review=re.sub("[^a-zA-z]"," ",review) 
    review=review.replace('*', ' stars')
    review=review.lower() 
    review=word_tokenize(review)
    review=[lemmatizer.lemmatize(word) for word in review if word not in stop_words]
    review=[word for word in review if len(word) > 1]
    review=" ".join(review)
            
    return review

@st.cache()
def load_data():
    path = Path('tripadvisor')
    df1 = pd.read_json(path/'train.unique.json', lines=True)
    df2 = pd.read_json(path/'test.unique.json', lines=True)
    df3 = pd.read_json(path/'test.zero.json', lines=True)
    df4 = pd.read_json(path/'test.one.json', lines=True)
    df5 = pd.read_json(path/'test.two.json', lines=True)

    df = pd.concat([df1, df2, df3, df4, df5], axis=0)
    df.reset_index(inplace=True, drop=True)
    df['review'] = df.segments.str.join(' ')
    df = df[['annotatorId', 'hotelId', 'date', 'author', 'review', 'ratingOverall', 
            'ratingRoom', 'ratingLocation', 'ratingCleanliness', 'ratingService', 
            'ratingBusiness', 'ratingValue', 'ratingCheckin']]
    df['review_p'] = df.review.apply(process_review).values
    df['sentiment'] = np.where(df.ratingOverall > 3, 'Positive', 
                           np.where(df.ratingOverall == 3, 'Neutral', 'Negative'))
    return df

st.markdown('# Hotel Review Analysis')

tab = st.sidebar.selectbox('Select Task', 
                ('Review Analysis', 'Aggregated Stats', 'Aspect Base Sentiment Analysis'))

if tab == 'Review Analysis':
    st.markdown('## Review analysis')
    review = st.text_input('Please type your review')
    

    if review:        
        # st.markdown('## Name Entity Recognition (NER)')
        docx = nlp(review)
        spacy_streamlit.visualize_ner(docx,
                                    show_table=False,
                                    labels=nlp.get_pipe('ner').labels)

        col1, _, col2, _, col3 = st.beta_columns([8, 1, 6, 1, 6])
        col1.markdown('### Processes text')
        review_p = process_review(review)
        col1.markdown(review_p)

        ## Emotional Sentiment 
        col2.markdown('### Emotions')
        col2.table(pd.Series(NRCLex(review).raw_emotion_scores))

        col3.markdown('### Sentiment')
        blob = TextBlob(review)
        col3.markdown('**Note:** Positive = 1 & Negative = 0')
        col3.markdown(f'Polarity : {blob.sentiment.polarity:0.2f}')
        with col3.beta_expander('Looks like a wrong prediction?'):
            correct = st.selectbox('Select the correct label',
                                    ('None', 'Positive', 'Neutral', 'Negative'))
            if correct != 'None':
                st.balloons()
            
elif tab == 'Aggregated Stats':
    st.markdown('## Aggregated Stats')
    df = load_data()
    
    hotelId = st.multiselect('Hotel Id', list(df.hotelId.unique()))
    col11, col12, col13 = st.beta_columns(3)
    rating = col11.multiselect('Rating', list(df.ratingOverall.unique()))
    sentiment = col12.multiselect('Sentiment', list(df.sentiment.unique()))
    n_words = col13.slider('Number of words', 5, 30) 

    c1 = df.ratingOverall.isin(rating) if rating    else True
    c2 = df.hotelId.isin(hotelId)      if hotelId   else True
    c3 = df.sentiment.isin(sentiment)  if sentiment else True
    c = c1 & c2 & c3

    count = Counter()

    if isinstance(c, bool):
        reviews = df.review_p
        rating = df.ratingOverall
        sentiment = df.sentiment
    else:
        reviews = df.loc[c, 'review_p']
        rating = df.loc[c, 'ratingOverall']
        sentiment = df.loc[c, 'sentiment']

    for text in reviews:
        for word in text.split():
            count[word] = count[word] + 1


    col30, _, col31, _, col32 = st.beta_columns([5, 1, 5, 1, 5])
    col30.subheader('Word Frequency')
    tmp = pd.DataFrame(count.most_common(n_words), columns=['Word', 'Frequency'])
    col30.table(tmp)
    
    col31.subheader('Ratings')
    col31.table(rating.value_counts().sort_index(ascending=False))
    
    col32.subheader('Sentiment')
    col32.table(sentiment.value_counts())

    s = None
    for text in reviews:
        if s is None:
            s = pd.Series(NRCLex(text).raw_emotion_scores)
        else:
            s += pd.Series(NRCLex(text).raw_emotion_scores)
        s = s.fillna(0)
    if reviews.any():
        st.subheader('Emotions')
        st.bar_chart(s.astype(int))


if tab == 'Aspect Base Sentiment Analysis':
    st.markdown('## Aspect Base Sentiment Analysis')
