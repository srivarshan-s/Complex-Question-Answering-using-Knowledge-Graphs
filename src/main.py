# Importing Libraries

from init import pd, nlp, tqdm, nx, plt
from entity import get_entities, get_relation

ROOT_PATH = '/run/media/sri/OS/Users/sriva/Desktop/Sem 5/'
ROOT_PATH += 'Knowledge Representation/Project/'
ROOT_PATH += 'Complex-Question-Answering-using-Knowledge-Graphs/'


# Read the data

candidate_sentences = pd.read_csv(ROOT_PATH + "data/wiki_sentences_v2.csv")
candidate_sentences = candidate_sentences.head(200)

entity_pairs = []
relations = []

print("Loading Data...")
for i in tqdm(candidate_sentences["sentence"]):
  entity_pairs.append(get_entities(i))
  relations.append(get_relation(i))

# extract subject
source = [i[0] for i in entity_pairs]

# extract object
target = [i[1] for i in entity_pairs]

kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})

print("Enter your question:")
question = input()
subject, _ = get_entities(question)
edge = get_relation(question)

for i in range(kg_df.shape[0]):
    if kg_df.source[i] == subject and kg_df.edge[i] == edge:
        print(kg_df.source[i], edge, kg_df.target[i])

# Plot Graph
G = nx.from_pandas_edgelist(kg_df[kg_df['edge']==edge], "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())
plt.figure(figsize=(12,12))
pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes
nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
plt.savefig(ROOT_PATH + "graph/" + edge + "_graph.png")
