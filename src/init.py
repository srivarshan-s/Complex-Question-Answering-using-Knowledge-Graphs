# Importing Libraries

import re
import pandas as pd
import bs4
import requests
import spacy

from spacy import displacy
nlp = spacy.load('en_core_web_sm')

from spacy.matcher import Matcher
from spacy.tokens import Span

import networkx as nx

import matplotlib.pyplot as plt

from tqdm import tqdm


# Set root project path

ROOT_PATH = '/run/media/sri/OS/Users/sriva/Desktop/Sem 5/Knowledge Representation/Project/knowledge-graph/'