from nltk.corpus import wordnet
from itertools import combinations, product

words = []

with open("violence.txt", 'r') as file:
  for line in file.readlines():
    words.append(line.strip())
# Collect violence words

sum = 0.0
count_ignore = 0
count = 0
for first, second in combinations(words, 2):
  first_synsets = wordnet.synsets(first)
  second_synsets = wordnet.synsets(second)
  path_sim_values = [wordnet.path_similarity(first_synset, second_synset) for first_synset, second_synset in product(first_synsets, second_synsets)]
  if len(path_sim_values) == 0:
    count += 1
    continue
  sum += max(path_sim_values)
  count += 1
  count_ignore += 1
  

print("Average : ", sum / count)
print("Average : ", sum / count_ignore)