#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:37:40 2019

@author: Marta
"""

import requests
import re
import time 
import string
import pandas as pd 

url = "https://www.lyrics.com/artist/Tupac-Shakur/557759"
page = requests.get(url)
text = page.text

pattern = "href=\"(/lyric/\d+/2Pac/[^\"]+)" 
links = re.findall(pattern, text, re.IGNORECASE)
base = "https://www.lyrics.com"

### create html files from all song lyric pages ###
for l in links:
    songlink = base + l
    songpage = requests.get(songlink)
    htmltext = songpage.text
    name = songlink.find("2Pac") #to name the file after the songname find where the word 2pac in the list starts
    #because that reoccurs in all songlinks
    songname = songlink[name+5:] #the name of the song starts 5 characters after the name "2Pac" (the lenght of 2Pac + /)
    transtable = str.maketrans('', '', string.punctuation) #replace all punctuation with whitespace
    songname = songname.translate(transtable)
    file = open(f"{songname}.txt", "w") #creates files with the name of the song
    file.write(htmltext)
    file.close()
    time.sleep(7)



### extract text from the xtml files ###
#create a list of all files in the directory that end with .txt (song lyric files)
import os
songfiles = []
for root, dirs, files in os.walk('/Users/Marta/Desktop/lyrics'):
    for file in files:
        if file.endswith('.txt'):
            songfiles.append(file)

from bs4 import BeautifulSoup
alllyrics = []  #create a list of strings that contain the lyrics for all songs         
for i in songfiles:
    html= open(f"{i}").read()
    soup = BeautifulSoup(html)
    txtpart = soup.find('pre') #find an ID (that is unique for each element) that is at the start of the lyrics text
    lyrics = txtpart.get_text()
    alllyrics.append(lyrics)



### Create a bag of words ###
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

count_vect = CountVectorizer(strip_accents = False, stop_words = "english",
                             min_df = 3)
counts = count_vect.fit_transform(alllyrics)
counts.shape
absolutefreq = counts.todense()
absolutefreq_vocab = count_vect.vocabulary_ 
#create a dataframe out of the dictionary
df_words = pd.DataFrame.from_dict(absolutefreq_vocab, orient='index')


tf_transformer = TfidfTransformer(use_idf=False).fit(counts)
termfreq = tf_transformer.transform(counts)
termfreq.shape
relativefreq = termfreq.todense()


#a combination of the both above
#TfidfVect = TfidfVectorizer(strip_accents = ascii, stop_words = {"english"},
#                             min_df = 3)
#tfcounts = TfidfVect.fit_transform(alllyrics)
#counts.shape
#relativefreq = tfcounts.todense()
#absolutefreq_vocab = TfidfVect.vocabulary_ 








#### EMINEM ####














