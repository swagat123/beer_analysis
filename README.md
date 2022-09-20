# Beer Data Analysis

I have used the data from the below link. The Source data is in csv format.

[https://drive.google.com/open?id=1e-kyoB97a5tnE7X4T4Es4FHi4g6Trefq]

I have run this code using google colab . To run this code you need to replace the below section in the code to point to your  path where you will have the source csv file (BeerDataScienceProject.csv).

```
from google.colab import drive
drive.mount('/content/drive')
basepath = "/content/drive/My Drive/Colab Notebooks/beer_analysis/"
file = os.path.join(basepath,"BeerDataScienceProject.csv")
```

## Purpose
Goal is to address queries like top Breweries which produce the strongest beers, which factor contribute most is it taste, aroma, appearance, or palette based on over all rating. If you want to recommend beer to your friend then which one you would like to recommend. Which beer style seems to be favourite based on user review comments etc.

## Dataset 
The data set is having  0.52887 million reviews data.Below listed fields are in the data set.
* beer_ABV	
* beer_beerId	
* beer_brewerId
* beer_name	
* beer_style	
* review_appearance	
* review_palette	
* review_overall	
* review_taste	
* review_profileName	
* review_aroma	
* review_text	
* review_time


## Python libraries 
I have used the below libraries 

```
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.mlab as mlab
import datetime 
import re


import os

import warnings
warnings.filterwarnings(action = 'ignore')

import nltk
from nltk.corpus import stopwords
from textblob import Word, TextBlob

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = stopwords.words('english')
nltk.download('omw-1.4')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('vader_lexicon')

lem = WordNetLemmatizer()

```

##Feature work
* We can include few feature engineering concepts using review rating's  create weighted review.
* Analyze how the data is dsitributed in review_overall,review_aroma,review_appearance,review_palate,review_taste, try to normalize the skewed data.
* Additional EDA to know more about the data.
