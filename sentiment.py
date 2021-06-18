from nltk.corpus import brown
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
cat_dict = dict()
for cat in brown.categories():
  sents = brown.sents(categories=cat)
  compound_score_sum = 0.0
  neg_score_sum = 0.0
  neutral_score_sum = 0.0
  pos_score_sum = 0.0
  for sent in sents:
    sent = " ".join(sent)
    ss = sid.polarity_scores(sent)
    compound_score_sum += ss["compound"]
    neg_score_sum += ss["neg"]
    neutral_score_sum += ss["neu"]
    pos_score_sum += ss["pos"]
    if ss["pos"] > 0.8:
      print("POSITIVE!!! : ", cat, sent)
    if ss["neg"] > 0.8:
      print("NEGATIVE!!! : ", cat, sent)
  cat_dict[cat, "compound"] = compound_score_sum / len(sents)
  cat_dict[cat, "neg"] = neg_score_sum / len(sents)
  cat_dict[cat, "neu"] = neutral_score_sum / len(sents)
  cat_dict[cat, "pos"] = pos_score_sum / len(sents)
for k in ["news", "reviews", "fiction", "mystery", "science_fiction", "romance", "humor"]:
  print("{} : {:.3%} {:.3%} {:.3%} {:.3%}".format(k, cat_dict[k, "compound"], cat_dict[k, "neg"], cat_dict[k, "neu"], cat_dict[k, "pos"]))