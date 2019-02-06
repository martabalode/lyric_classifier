# lyric_classifier

otherartists.py
1. Scrapes lyrics.com for 3 artists (Eminem, Dr. Dre, Ice Cube)
2. Extracts HTML files from the lyrics pages
3. Extracts text from the HTML files using BeautifulSoup
4. Builds a text corpus
5. Implements text vectorization (Tf-Id-Vectorization)
6. Constructs a Naive Bayes model that predicts the artist of a song given new song lyrics
7. Scrapes lyrics.com for 1 artist (Tupac) --> tupac.py
2. Applies the NB model to the new lyrics
