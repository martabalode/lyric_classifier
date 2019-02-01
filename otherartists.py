#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 09:37:19 2019

@author: Marta
"""

### other artists ###
import re
from collections import Counter
import requests
import time
import os
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

base = "https://www.lyrics.com"
otherartists = ["/artist/Eminem/347307", "/artist/Dr.-Dre/26119", "/artist/Ice-Cube/48"]

############################

def getsonglinks(artist):

    artistlink = base + artist
    artistpage = requests.get(artistlink)
    pagetext = artistpage.text

    pattern = "href=\"(/lyric/\d+/[^/]+/[^\"]+)"  
    links = re.findall(pattern, pagetext, re.IGNORECASE)
    
    arnames = []
    for i in links:
        arname = i.split(sep = "/")[3]
        arnames.append(arname)
    
    z = list(zip(links, arnames))
    
    allarnamesdict = Counter(arnames)
    mostpopular = allarnamesdict.most_common(1)
    x, y = mostpopular[0]
   
    cleanedlinks = []
    for link, name in z:
        if name == x:
            cleanedlinks.append(link)
    
    return cleanedlinks

############################

listofartistsonglinks = []
for i in otherartists:
    linklist = getsonglinks(i)
    listofartistsonglinks.append(linklist)

############################



def gethtmlfromlinks(linklist):
    
    for link in linklist:
        #call each song link
        songlink = base + link
        songpage = requests.get(songlink)
        htmltext = songpage.text
        #create a file name out of the link
        artistname = link.split(sep = "/")[3]
        startnr = link.find(artistname)
        songname = link[startnr+len(artistname):]
        filename = artistname + songname
        filename = re.sub("/", "__", filename) #cant include / because thinks that its a folder name
        #write a html file
        file = open(f"{filename}.txt", "w")
        file.write(htmltext)
        file.close()
        time.sleep(7)

gethtmlfromlinks(listofartistsonglinks[2])


############################

for linklist in listofartistsonglinks:
    gethtmlfromlinks(linklist)
    time.sleep(20)
    
############################

#get artistnames
artistnames = []
for list in listofartistsonglinks:
    for i, link in enumerate(list):
        if i == 0:
            artistname = link.split(sep = "/")[3]
            artistnames.append(artistname)

#get a list of lists with html file names for each artist
listofsongfilelists = []
for artistname in artistnames:
    oneartistssongfiles = []
    for root, dirs, files in os.walk('/Users/Marta/Desktop/lyrics'):
        for file in files:
            if file.endswith('.txt'):
                if file.startswith(f"{artistname}"):
                    oneartistssongfiles.append(file)
    listofsongfilelists.append(oneartistssongfiles)

#extract text from html files
allartistlyrics = [] 
for list in  listofsongfilelists:
    oneartistlyrics = []
    for songfile in list:
        try:
            html= open(f"{songfile}").read()
            soup = BeautifulSoup(html)
            txtpart = soup.find('pre') 
            lyrics = txtpart.get_text()
            oneartistlyrics.append(lyrics)
        except AttributeError:
            print(f"skipping {songfile}")
            continue
    allartistlyrics.append(oneartistlyrics)

############################
    
def makelabels(allartistlyrics):
    labelslist = []
    for i, oneartistlyrics in enumerate(allartistlyrics):
        name = artistnames[i]
        labels = [name for j in range(len(oneartistlyrics))]
        labelslist.append(labels)
    return labelslist

labelslist = makelabels(allartistlyrics)


def flattenlist(everybiglist):
    flatlist = [i for sublist in everybiglist for i in sublist]
    return flatlist
    
flat_allartistlyrics = flattenlist(allartistlyrics)   
flat_labelslist = flattenlist(labelslist) 
    
############################

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
cv = CountVectorizer(stop_words = "english")
counts = cv.fit_transform(flat_allartistlyrics)
densecounts = counts.todense()
cv.vocabulary_ 


TfidfVect = TfidfVectorizer(strip_accents = ascii, stop_words = {"english"},
                             min_df = 3)
tfcounts = TfidfVect.fit_transform(flat_allartistlyrics) #not showing in variable explorer, because compressed
uniquenrtoeachword = TfidfVect.vocabulary_ 
todense = tfcounts.todense() #to view the lenght/size of the sparse matrix


X = tfcounts
y = flat_labelslist

from sklearn.naive_bayes import MultinomialNB
m = MultinomialNB()
m.fit(X, y)
m.score(X, y)
predicted_y = m.predict(X)
predictedprob_y = m.predict_proba(X)

############################
# apply to new data
test_songs = ["aaaaaa", 
              "i dont wanna be on top of the mountain", 
              "say what give me one more time",
              "residual autocorrelation can indicate many types of misspecification of a model"]

xtest = TfidfVect.transform(test_songs)
m.predict(xtest)

############################

##correct the Eminem bias = make the lists the same lenght##

def getlistlenghts(listoflists):
    listlenghts = []
    for i in range(len(listoflists)):
        listlenght = len(listoflists[i])
        listlenghts.append(listlenght)
    return listlenghts
    
# Make the lists match the shorter list #
listlenghts = getlistlenghts(allartistlyrics)
smallest = min(listlenghts)
reducedallartistlyrics = [x[0:smallest] for x in allartistlyrics]


reducedlabelslist = makelabels(reducedallartistlyrics) 

# Make the lists match the longer list #
listlenghts = getlistlenghts(allartistlyrics)
largest = max(listlenghts)

increasedallartistlyrics = []
for artistlist in allartistlyrics:
    differencefromlargest = largest - len(artistlist)
    if differencefromlargest == 0:
        increasedallartistlyrics.append(artistlist)
    else:
        slicedlist = artistlist[: differencefromlargest]
        increasedlist = artistlist + slicedlist
        increasedallartistlyrics.append(increasedlist)

increasedlabelslist = makelabels(increasedallartistlyrics)

#### apply the NB multinomial models to both the reduced and the increased list ####

## to the reduced list ##
flat_reducedallartistlyrics = flattenlist(reducedallartistlyrics)   
flat_reducedlabelslist = flattenlist(reducedlabelslist) 

TfidfVect = TfidfVectorizer(strip_accents = ascii, stop_words = {"english"},
                             min_df = 3)
tfreduced = TfidfVect.fit_transform(flat_reducedallartistlyrics) 
Xreduced = tfreduced
yreduced = flat_reducedlabelslist
m = MultinomialNB()
m.fit(Xreduced, yreduced)
m.score(Xreduced, yreduced)

## to the increased list ##
flat_increasedallartistlyrics = flattenlist(increasedallartistlyrics)   
flat_increasedlabelslist = flattenlist(increasedlabelslist) 

TfidfVect = TfidfVectorizer(strip_accents = ascii, stop_words = {"english"},
                             min_df = 3)
tfincreased = TfidfVect.fit_transform(flat_increasedallartistlyrics) 
Xincreased = tfincreased
yincreased = flat_increasedlabelslist
m = MultinomialNB()
m.fit(Xincreased, yincreased)
m.score(Xincreased, yincreased)


## apply to new data ##
test_songs = ["aaaaaa", 
              "eight mile detroit slim shady", 
              "say what give me one more time",
              "residual autocorrelation can indicate many types of misspecification of a model"]

#reduced prediction#
TfidfVect = TfidfVectorizer(strip_accents = ascii, stop_words = {"english"},
                             min_df = 3)
tfreduced = TfidfVect.fit_transform(flat_reducedallartistlyrics) 
Xreduced = tfreduced
yreduced = flat_reducedlabelslist
m = MultinomialNB()
m.fit(Xreduced, yreduced)
xtest = TfidfVect.transform(test_songs)
reducedprediction = m.predict(xtest)

#increased prediction#
TfidfVect = TfidfVectorizer(strip_accents = ascii, stop_words = {"english"},
                             min_df = 3)
tfincreased = TfidfVect.fit_transform(flat_increasedallartistlyrics) 
Xincreased = tfincreased
yincreased = flat_increasedlabelslist
m = MultinomialNB()
m.fit(Xincreased, yincreased)
xtest = TfidfVect.transform(test_songs)
increasedprediction = m.predict(xtest)


### apply on Tupac ###
TfidfVect = TfidfVectorizer(strip_accents = ascii, stop_words = {"english"},
                             min_df = 3)
tfincreased = TfidfVect.fit_transform(flat_increasedallartistlyrics) 
Xincreased = tfincreased
yincreased = flat_increasedlabelslist
m = MultinomialNB()
m.fit(Xincreased, yincreased)
xtest = TfidfVect.transform(alllyrics)
increasedprediction = m.predict(xtest)














