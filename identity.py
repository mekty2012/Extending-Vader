from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist, ConditionalFreqDist

stop_words = stopwords.words()

lines = []
freqdist = ConditionalFreqDist()
with open("../all_data.csv", encoding='UTF8') as file:
  index = file.readline().strip("\n").split(",")

  while True:
    line = file.readline()
    if not line:
      break
    if line.count(",") != 45:
      continue
    
    values = line.strip("\n").split(",")
    max_id = max([(i + 20, float(v) if v else 0.0) for i, v in enumerate(values[20:44])], key=lambda t : t[1])
    if max_id[1] > 0.5:
      for word in word_tokenize(values[1]):
        if word.lower() not in stop_words and word.isalpha():
          freqdist[index[max_id[0]]][word.lower()] += 1

identities = []
for cat in freqdist:
  count = 200
  for word in freqdist[cat].most_common(200):
    print(cat, word)
    tf = input("%d left" % count)
    if len(tf) > 0:
      identities.append(word)
    count -= 1
print(identities)