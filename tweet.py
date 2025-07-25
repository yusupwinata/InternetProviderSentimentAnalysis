import os
import re
import string
import pandas as pd
from nlp_id.lemmatizer import Lemmatizer
from nlp_id.stopword import StopWord

ABSOLUTE_PATH = os.path.abspath(os.getcwd())
CUSTOM_SLANGS = os.path.join(ABSOLUTE_PATH, "dataset", "custom_slangs.csv")
CUSTOM_NON_STOPWORDS = os.path.join(ABSOLUTE_PATH, "dataset", "custom_non_stopwords.csv")

class Tweet():
    """
    A class used to represent an tweet text
    """
    lemmatizer = Lemmatizer()
    lemmatizer.root_word.add("ponsel")
    lemmatizer.root_word.add("spam")

    stopword = StopWord()

    @staticmethod
    def _load_custom_slangs() -> dict:
        """Load custom slangs from local csv file"""
        try:
            df = pd.read_csv(CUSTOM_SLANGS)
            slangs_dict = dict(zip(df["Slang"], df["Formal_Lang"]))
            return slangs_dict
        except Exception as e:
            raise Exception(f"Error loading custom slangs: {str(e)}")
        
    @staticmethod
    def _load_custom_non_stopwords() -> set:
        """Load custom non-stop words from local csv file"""
        try:
            df = pd.read_csv(CUSTOM_NON_STOPWORDS)
            non_stopwords = set(df["Non_Stopword"])
            return non_stopwords
        except Exception as e:
            raise Exception(f"Error loading custom non-stop words: {str(e)}")
    
    slangs_dict = _load_custom_slangs()
    non_stopwords = _load_custom_non_stopwords()

    def __init__(self, original_tweet: str, sentiment: str):
        self.original_tweet = original_tweet
        self.sentiment = sentiment
        self.clean_tweet, self.hashtags = self.preprocessing(self.original_tweet)
    
    def convert_slangs(self, word: str) -> str:
        """Convert provided slang to appropriate term"""
        return self.slangs_dict.get(word, word)
    
    def remove_stopword(self, word: str) -> str:
        """Remove stop word from the provided text"""
        if word not in self.non_stopwords:
            return self.stopword.remove_stopword(word)
        return word
    
    def preprocessing(self, text: str) -> tuple:
        """Perform preprocessing on the provided tweet"""
        # Remove twitter tags
        text = re.sub(r"\bRT\b", "", text)
        text = re.sub(r"<(.*?)>", "", text)
        text = text.replace("…", "")

        # Convert emoticons to appropriate terms
        text = re.sub(r"(:[v|V])", "cengang", text)
        text = re.sub(r"(:\)\))|(:\)|:D)", "senyum", text)
        text = re.sub(r"(XD)", "tawa", text)

        # Lower casing
        text = text.strip().lower()

        tweet = []
        hashtags = []
        for word in text.split():
            # Retain hastags
            if len(word) > 1 and word[0] == "#":
                hyphens = ["_"]
                if word[-1] in hyphens:
                    tweet.append(word[:-1])
                    hashtags.append(word[:-1])
                else:
                    tweet.append(word)
                    hashtags.append(word)
            else:
                # Remove special characters at the end the word
                clean_word = word
                while len(clean_word)>=1 and clean_word[-1] in string.punctuation:
                    if clean_word == "h+": # retain specific terms
                        break
                    clean_word = clean_word[:-1]
                
                # Convert slangs
                clean_word = self.convert_slangs(clean_word)

                # Remove the prefix "-nya"
                if len(clean_word)>=4 and clean_word[-4:]=="-nya":
                    clean_word = clean_word[:-4]
                if len(clean_word)>=3 and clean_word[-3:]=="nya":
                    clean_word = clean_word[:-3]

                # Handle "kata ulang" except digits
                if len(clean_word)>=1 and clean_word[-1] == "2":
                    kata_awal = clean_word[:-1]
                    if not bool(re.search(r"\d", kata_awal)):
                        if kata_awal in self.lemmatizer.root_word:
                            clean_word = kata_awal
                        elif self.convert_slangs(kata_awal) and self.convert_slangs(kata_awal) in self.lemmatizer.root_word:
                            clean_word = self.convert_slangs(kata_awal)
                        elif self.lemmatizer.lemmatize(kata_awal):
                            clean_word = self.lemmatizer.lemmatize(kata_awal)
                
                # Retain digits
                if not bool(re.search(r"\d", clean_word)):
                    # Retain specific terms
                    if clean_word == "h+":
                        pass
                    else:
                        # Lemmatization
                        clean_word = self.lemmatizer.lemmatize(clean_word)

                        # Stop word removal
                        clean_word = self.remove_stopword(clean_word)
                
                if clean_word:
                    tweet.append(clean_word)
        
        # Return clean text tweet
        tweet = " ".join(tweet)
        return tweet, hashtags