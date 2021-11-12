# Importing Libraries
from init import pd, ROOT_PATH
from knowledge_graph import load_data, plot_graph
from knowledge_graph import extract_sub_obj, get_question, answer


# Read the data
candidate_sentences = pd.read_csv(ROOT_PATH + "data/wiki_sentences_v2.csv")
candidate_sentences = candidate_sentences.head(200)


# Load entity pairs and relations
print("Loading Data...")
entity_pairs, relations = load_data(
        candidate_sentences["sentence"])


# Extract subject, object
source, target = extract_sub_obj(entity_pairs)


# Create knowledge graph dataframe
kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})


# Get question from user
subject, edge = get_question()


# Obtain answer from knowledge graph
answer(kg_df, subject, edge)


# Plot knowledge graph
plot_graph(kg_df, edge)
