from nltk.corpus import wordnet

violent_words = []
violent_synsets = dict()
collection = dict()

with open("violence.txt", 'r') as file:
  for line in file.readlines():
    word = line.strip()
    synsets = wordnet.synsets(word)
    if len(synsets) == 0:
      continue
    violent_words.append(word)
    collection[word] = []
    violent_synsets[word] = wordnet.synsets(word)
    

for ss in wordnet.all_synsets('v'):
  path_similarities = [(word, max([wordnet.path_similarity(ss, v_ss) for v_ss in violent_synsets[word]])) for word in violent_words]
  max_item = max(path_similarities, key=lambda t : t[1])
  if max_item[1] > 0.5:
    collection[max_item[0]].extend([lemma.name() for lemma in ss.lemmas()])

with open("extended_violent.csv", 'w') as file:
  for key in collection:
    for word in collection[key]:
      file.write("{},{}\n".format(key, word))

print(collection)