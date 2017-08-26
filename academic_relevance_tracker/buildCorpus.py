from urllib import request
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import string

url = "http://www.raid2015.org/review.html"
tokensstart = 152
tokensend = 7626

''' Academic stopwords to add:
 paper
 pc
 authors
 work
 reviewers
'''

class AcademicCorpus (object):
    ''' Represents an NLTK wrapped object for conference programs. '''
    def _init_(self, url):
        ''' Initializes the resource (e.g. conference) from the given URL '''
        self.url = url
        self.text = _textFromURL(url)

    def getText(self):
        return nltk.pos_tag(self.text)

    def getTaggedText(self):
        return

    def _textFromURL(url, tokensstart, tokensend = None):
        # Get the page source and scan it for text:
        html = request.urlopen(url).read().decode('utf8')
        raw = BeautifulSoup(html).get_text()

        # Case-normalize
        lowered = raw.lower()

        # Remove punctuation
        no_punctuation = lowered.translate(str.maketrans('','', string.punctuation))

        # Tokenize, strip off identified unnecessary tokens (e.g. page tokens, etc...),
        # and remove stopwords -- pruning code should be automatable w/ deep learning?
        tokens = word_tokenize(no_punctuation)
        if not tokensend:
            tokensend = len(tokens)
        tokens = tokens[tokensstart:tokensend]
        filtered = [w for w in tokens if not w in stopwords.words('english')]

        # Stem the tokens:
        stemmer = PorterStemmer()
        stemmed = _stem_tokens(filtered, stemmer)

        # Initialize the text
        self.text = nltk.Text(stemmed)

    def _stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

if __name__ == "__main__":
    '''
    Initialize the class
    '''
