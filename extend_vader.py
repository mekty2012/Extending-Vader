from nltk import tokenize
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from datetime import datetime
import sys

sid = SentimentIntensityAnalyzer()
stemmer = WordNetLemmatizer()

with open("extend_violent.csv", 'r') as file:
  for line in file.readlines():
    line = line.strip().split(",")
    sid.lexicon[line] = min(-3.0, sid.lexicon[line] if line in sid.lexicon else -3.0)

with open("violence.txt", 'r') as file:
  for line in file.readlines():
    line = line.strip()
    sid.lexicon[line] = min(-4.0, sid.lexicon[line] if line in sid.lexicon else -4.0)

with open("identity.txt", "r"):
  for line in file.readlines():
    line = line.strip()
    sid.constants.BOOSTER_DICT[line] = sid.constants.B_INCR

text = "\n".join(open(sys.argv[1]).readlines())

def lemmatize_sent(sent):
  words = pos_tag(tokenize.word_tokenize(sent))
  res = []
  for w, p in words:
    try:
      word = stemmer.lemmatize(w, p)
    except:
      word = w
    res.append(word)
  return " ".join(res)

sents = tokenize.sent_tokenize(text)
begin = datetime.today()
for sent in sents:
  print(sid.polarity_scores(lemmatize_sent(sent)))
print((datetime.today() - begin).microseconds)