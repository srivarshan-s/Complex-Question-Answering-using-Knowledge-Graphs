from init import nlp, Matcher, tqdm, plt, nx, ROOT_PATH


def load_data(sentences):
    
    entity_pairs = []
    relations = []

    for i in tqdm(sentences):
        entity_pairs.append(get_entities(i))
        relations.append(get_relation(i))

    return entity_pairs, relations


def extract_sub_obj(entity_pairs):

    subject = [ i[0] for i in entity_pairs ]
    target = [ i[1] for i in entity_pairs ]

    return subject, target


def get_question():

    print("Enter your question:")
    question = input()
    subject, _ = get_entities(question)
    edge = get_relation(question)

    return subject, edge


def answer(dataframe, subject, edge):

    for i in range(dataframe.shape[0]):
        if dataframe.source[i] == subject and dataframe.edge[i] == edge:
            print(dataframe.source[i], edge, dataframe.target[i])


def plot_graph(dataframe, relation):

    G = nx.from_pandas_edgelist(dataframe[dataframe['edge']==relation], "source", "target", 
                              edge_attr=True, create_using=nx.MultiDiGraph())
    plt.figure(figsize=(12,12))
    pos = nx.spring_layout(G, k = 0.5) 
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
    plt.savefig(ROOT_PATH + "graph/" + relation + "_graph.png")


def get_entities(sent):

  ent1 = ""
  ent2 = ""

  prv_tok_dep = ""    
  prv_tok_text = ""  

  prefix = ""
  modifier = ""

  for tok in nlp(sent):

    if tok.dep_ != "punct":

      if tok.dep_ == "compound":
        prefix = tok.text

        if prv_tok_dep == "compound":
          prefix = prv_tok_text + " "+ tok.text

      if tok.dep_.endswith("mod") == True:
        modifier = tok.text

        if prv_tok_dep == "compound":
          modifier = prv_tok_text + " "+ tok.text
      
      if tok.dep_.find("subj") == True:
        ent1 = modifier +" "+ prefix + " "+ tok.text
        prefix = ""
        modifier = ""
        prv_tok_dep = ""
        prv_tok_text = ""      

      if tok.dep_.find("obj") == True:
        ent2 = modifier +" "+ prefix +" "+ tok.text
        
      prv_tok_dep = tok.dep_
      prv_tok_text = tok.text

  return [ent1.strip(), ent2.strip()]


def get_relation(sent):

  doc = nlp(sent)

  matcher = Matcher(nlp.vocab)

  pattern = [{'DEP':'ROOT'}, 
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},  
            {'POS':'ADJ','OP':"?"}] 

  matcher.add("matching_1", [pattern]) 

  matches = matcher(doc)
  k = len(matches) - 1

  if k == -1:
    return " "

  else:
    span = doc[matches[k][1]:matches[k][2]] 

  return(span.text)
