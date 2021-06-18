from nltk import tokenize
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

sid = SentimentIntensityAnalyzer()
baseline_sid = SentimentIntensityAnalyzer()
stemmer = WordNetLemmatizer()

with open("extend_violent.csv", 'r') as file:
  for line in file.readlines():
    line = line.strip().split(",")
    sid.lexicon[line[1]] = min(-3.0, sid.lexicon[line[1]] if line[1] in sid.lexicon else -3.0)


with open("violence.txt", 'r') as file:
  for line in file.readlines():
    line = line.strip()
    sid.lexicon[line] = min(-4.0, sid.lexicon[line] if line in sid.lexicon else -4.0)

with open("identity.txt", "r") as file:
  for line in file.readlines():
    line = line.strip()
    sid.constants.BOOSTER_DICT[line] = sid.constants.B_INCR

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

result = dict()
count = 50000
with open("../all_data.csv", encoding='UTF8') as file:
  l = file.readline().strip("\n").split(",")
  for i in range(13, 20):
    result[l[i]] = [0.0, 0]
  result["wrong"] = [0.0, 0]
  
  while count > 0:
    line = file.readline()
    if not line:
      break
    if line.count(",") != 45:
      continue

    values = line.strip("\n").split(",")
    toxicity_values = values[13:20]
    max_toxicity = max([(i + 13, float(v) if v else 0.0) for i, v in enumerate(toxicity_values)], key=lambda t : t[1])
    count -= 1
    if max_toxicity[1] > 0.5:
      text = values[1]
      sentences = tokenize.sent_tokenize(text)
      for sentence in sentences:
        sentence = lemmatize_sent(sentence)
        new_score = sid.polarity_scores(sentence)['neg']
        bef_score = baseline_sid.polarity_scores(sentence)['neg']
        result[l[max_toxicity[0]]][0] += new_score - bef_score
        result[l[max_toxicity[0]]][1] += 1
    else:
      text = values[1]
      sentences = tokenize.sent_tokenize(text)
      for sentence in sentences:
        sentence = lemmatize_sent(sentence)
        new_score = sid.polarity_scores(sentence)['neg']
        bef_score = baseline_sid.polarity_scores(sentence)['neg']
        result["wrong"][0] += bef_score - new_score
        result["wrong"][1] += 1
  
  for k in result:
    if result[k][1] == 0:
      continue
    print(k, result[k][0] / result[k][1])
        