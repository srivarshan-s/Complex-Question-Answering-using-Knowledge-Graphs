# Importing Libraries

from init import pd, nlp, tqdm
from entity import get_entities, get_relation

pd.set_option('display.max_colwidth', 200)

ROOT_PATH = '/run/media/sri/OS/Users/sriva/Desktop/Sem 5/'
ROOT_PATH += 'Knowledge Representation/Project/'
ROOT_PATH += 'Complex-Question-Answering-using-Knowledge-Graphs/'


# Read the data

candidate_sentences = pd.read_csv(ROOT_PATH + "data/wiki_sentences_v2.csv")

"""
text = "the film had 200 patents"
doc = nlp(text)

print(text)

for tok in doc:
  print(tok.text, "=>", tok.dep_)

print(get_entities(text))
"""

entity_pairs = []

for i in tqdm(candidate_sentences["sentence"]):
  entity_pairs.append(get_entities(i))

print(entity_pairs[0:10])
