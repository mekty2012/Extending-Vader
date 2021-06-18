from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

stop_words = stopwords.words()

lines = []
freqdist = FreqDist()
with open("../all_data.csv", encoding='UTF8') as file:
  file.readline()

  while True:
    line = file.readline()
    if not line:
      break
    if line.count(",") != 45:
      continue
    
    values = line.strip("\n").split(",")
    if float(values[19]) > 0.5:
      # If 'thread'
      lines.append(values[1])
      # Add commented_text

for threat in lines:
  words = word_tokenize(threat)
  for word in words:
    if word.lower() not in stop_words and word.isalpha():
      freqdist[word.lower()] += 1

print(freqdist.most_common(1000))

thread_word = []
for common in freqdist.most_common(1000):
  print(common[0], common[1])
  tf = input()
  if len(tf) > 0:
    thread_word.append(common[0])
print(thread_word)
