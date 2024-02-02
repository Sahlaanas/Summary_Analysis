from django.shortcuts import render
from django.http import HttpResponse
import json
# import tensorflow as tf
import spacy
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import string
from nltk.util import ngrams
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')
from textstat import gunning_fog
from spellchecker import SpellChecker
from .models import history

from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

from dotenv import load_dotenv

# Load variables from .env
load_dotenv()
# Load the model

# You can now use the loaded model for predictions or further training

with open(r'C:\Users\user\Desktop\MyProject\Model\columns.json', 'r') as json_file:
    data = json.load(json_file)
    columns = data['data_columns']
    

model = joblib.load(r'C:\Users\user\Desktop\MyProject\Model\ml_model1.joblib')
vectorizer = joblib.load(r'C:\Users\user\Desktop\MyProject\Model\vectorizer1.joblib')

# model = tf.keras.models.load_model('C:\Users\user\Desktop\MyProject\Model\lstm_model.h5')

# vectorizer = TfidfVectorizer()

# functions

nlp = spacy.load("en_core_web_md")

def count_stopwords(text: str) -> int:
    stopwords_list = set(stopwords.words('english'))
    words = text.split()
    stopwords_count = sum(1 for word in words if word.lower() in stopwords_list)
    return stopwords_count

def count_punctuation(text: str) -> int:
    punctuation_set = set(string.punctuation)
    punctuation_count = sum(1 for char in text if char in punctuation_set)
    return punctuation_count

def collapse_dots(input):
    # Collapse sequential dots
    input = re.sub("\.+", ".", input)
    # Collapse dots separated by whitespaces
    all_collapsed = False
    while not all_collapsed:
        output = re.sub(r"\.(( )*)\.", ".", input)
        all_collapsed = input == output
        input = output
    return output

def clean_text(text:str, join_back=True):
  result = []
  sentences = sent_tokenize(text)
  sentences = [collapse_dots(sentence) for sentence in sentences]
  sentences = [sentence.strip() for sentence in sentences]

  for sentence in sentences:
    words = word_tokenize(sentence)

    words = [re.sub(r'\s+', ' ', word) for word in words]
    words = [re.sub(r'[^a-zA-Z\s]', '', word) for word in words]
    words = [re.sub(r'[^\w\s]', '', word) for word in words]
    words = [word.lower() for word in words]
    # words = [word for word in words if word not in stop_words]
    filtered_sentence = " ".join(words)
    result.append(filtered_sentence)

  return " ".join(result)

#configuring stopwords
stop_words = set(stopwords.words('english'))
stop_words.remove('not')

def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    return lemmatized_text

def remove_extra_space(text:str):
  text = re.sub(r'\s+', ' ', text)
  text = text.strip()
  return text

def count_total_words(text: str) -> int:
    words = text.split()
    total_words = len(words)
    return total_words

def count_syllables(word):
    # Simple syllable counting (an approximation)
    vowels = "aeiouAEIOU"
    syllables = sum(1 for char in word if char in vowels)
    return syllables

def automated_readability_indexs(text):

    doc = nlp(text)
    # Calculate the number of words
    word_count = len(doc)

    # Calculate the number of sentences
    sentence_count = len(list(doc.sents))

    # Calculate the average number of letters per word
    letters_per_word = sum(len(word.text) for word in doc) / word_count

    # Calculate the average number of words per sentence
    words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0

    # Calculate the ARI
    ari = 4.71 * letters_per_word + 0.5 * words_per_sentence - 21.43

    return ari

def gunning_fog_index(text):
  text = text
  return gunning_fog(text)


# Function to count verbs
def count_verbs(text):
    doc = nlp(text)
    verbs = [token for token in doc if token.pos_ == "VERB"]
    return len(verbs)

# Function to count adjectives
def count_adjectives(text):
    doc = nlp(text)
    adjectives = [token for token in doc if token.pos_ == "ADJ"]
    return len(adjectives)

# Function to count adverbs
def count_adverbs(text):
    doc = nlp(text)
    adverbs = [token for token in doc if token.pos_ == "ADV"]
    return len(adverbs)

# Function to count nouns
def count_nouns(text):
    doc = nlp(text)
    nouns = [token for token in doc if token.pos_ == "NOUN"]
    return len(nouns)

# Function to count pronouns
def count_pronouns(text):
    doc = nlp(text)
    pronouns = [token for token in doc if token.pos_ == "PRON"]
    return len(pronouns)

def count_specific_pos(text, pos_tag):
    # Count the number of words with a specific part-of-speech tag (e.g., "ADP" for prepositions)
    doc = nlp(text)
    pos_count = sum(1 for token in doc if token.pos_ == pos_tag)
    return pos_count

def count_bigrams(text: str) -> int:
    words = word_tokenize(text)
    bigrams = list(ngrams(words, 2))
    return len(bigrams)

def count_trigrams(text: str) -> int:
    words = word_tokenize(text)
    trigrams = list(ngrams(words, 3))
    return len(trigrams)

spell = SpellChecker()

def misspelled_cnt(text):
  spell = SpellChecker()
  misspelled = spell.unknown(text.split())
  count = len(misspelled)
  return count


def unique_count(text):
  doc = nlp(text)
  words = [token.text for token in doc]
  unique = len(set(words))
  return unique

def remove_stopword(text):
    text_tokens = word_tokenize(text)
    text_tokens = [word for word in text_tokens if word not in stop_words]
    return ' '.join(text_tokens)

def cosine_similarity_text_prompt(text, prompt_text):
    # Tokenize and remove stopwords

    # Create TF-IDF vectors for text and prompt_text
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text, prompt_text])

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

    return cosine_sim[0][0]

# tokenizer = tf.keras.preprocessing.text.Tokenizer()

import google.generativeai as genai
import os



os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_APIKEY')
genai.configure(api_key = os.environ['GOOGLE_API_KEY'])


genmodel = genai.GenerativeModel('gemini-pro')


# Create your views here.
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

embd_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def prediction(request):
    if request.method == 'POST':
        prompt = request.POST['prompttext']
        summary = request.POST['summarytext']
        prompt1 = prompt
        summary1 = summary
        
        sentences = [prompt,summary]
        sentence_embedding = embd_model.encode(sentences)
        cossim = cosine_similarity(sentence_embedding[0].reshape(1,-1),sentence_embedding[1].reshape(1,-1))[0][0]
        if cossim > 0.5:
            x = [0,0]
            y = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            text_cntstpwrd = count_stopwords(summary)
            text_cnt_punct = count_punctuation(summary)
            summary = clean_text(summary)
            prompt = clean_text(prompt)
            summary = remove_extra_space(summary)
            prompt = remove_extra_space(prompt)
            text_wordcnt = count_total_words(summary)
            text_len = len(summary)
            text_syllablecnt = count_syllables(summary)
            text_ARI = automated_readability_indexs(summary)
            text_GFI = gunning_fog_index(summary)
            text_cntverb = count_verbs(summary)
            text_adjtv = count_adjectives(summary)
            text_cntadvrb = count_adverbs(summary)
            text_cntnoun = count_nouns(summary)
            text_cntpronoun = count_pronouns(summary)
            text_cntspecificpos = count_specific_pos(summary,'ADP')
            text_bigram = count_bigrams(summary)
            text_trigram = count_trigrams(summary)
            text_misspelt = misspelled_cnt(summary)
            text_unique = unique_count(summary)
            summary = remove_stopword(summary)
            prompt = remove_stopword(prompt)
            summary = lemmatize_text(summary)
            prompt = lemmatize_text(prompt)
            cosine_similarities = cosine_similarity_text_prompt(summary,prompt)

            x[0] = prompt
            x[1] = summary
            y[0] = text_cntstpwrd
            y[1] = text_cnt_punct
            y[2] = text_wordcnt
            y[3] = text_len
            y[4] = text_syllablecnt
            y[5] = text_ARI
            y[6] = text_GFI
            y[7] = text_cntverb
            y[8] = text_adjtv
            y[9] = text_cntadvrb
            y[10] = text_cntnoun
            y[11] = text_cntpronoun
            y[12] = text_cntspecificpos
            y[13] = text_bigram
            y[14] = text_trigram
            y[15] = text_misspelt
            y[16] = text_unique
            y[17] = cosine_similarities

            x_list = [" ".join(map(str, x))]
            x_vctrsd = vectorizer.transform(x_list)
            x_array = x_vctrsd.toarray()
            y_array = np.array([y])
            mergedarray = np.hstack((x_array, y_array))

            print(type(x_array),type(y_array),"00000")

            pred =  model.predict(mergedarray)
            content_score = round(pred[0][0],3)
            wording_score = round(pred[0][1],3)
            input_prompt = """
                You are an expert in understanding english language. We will give a summary 
                and you will have to generate a small description in a single paragraph about the mistakes in that text"""
        
            response = genmodel.generate_content([input_prompt,summary])
    
    
            data_history = history(prompttext=prompt1,summarytext=summary1,content = content_score,wording=wording_score)
            data_history.save()
    
            search_history = history.objects.all().order_by('-id')[:5]
            data = {
                'content' : content_score,
                'wording' : wording_score,
                'responses' : response.text
                
            }
    
            return render(request,'index.html',{'datas':data,'history' : search_history})
        else:
            search_history = history.objects.all().order_by('-id')[:5]
            return render(request,'index.html',{'flag' : True,'history':search_history})
        
    search_history = history.objects.all().order_by('-id')[:5]
    return render(request,'index.html',{'history':search_history})  

# def show_history(request):
#     search_history = history.objects.all()
#     return render(request, 'index.html',{'history':search_history})
        
def home(request):
    return render(request,'home.html')
    

        

