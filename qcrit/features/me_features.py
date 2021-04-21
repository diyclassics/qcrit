# pylint: disable = missing-docstring
'''
Middle English features
'''

from ..textual_feature import textual_feature
import re
import nltk
from itertools import chain
import numpy as np

def preprocess(text):
    text = re.sub(r'\*\b.+?\b\*', ' ', text) # Remove 'text' between asterisks
    text = re.sub(r'\{.+?\}', ' ', text) # Remove 'text' between curly brackets

    punctuation ="\"#$%&\'()*,-/:;<=>@[\]^_`{|}~.?!«»"
    translator = str.maketrans({key: " " for key in punctuation})
    text = text.translate(translator)

    punctuation ="+"
    translator = str.maketrans({key: "" for key in punctuation})
    text = text.translate(translator)

    translator = str.maketrans({key: " " for key in '0123456789'})
    text = text.translate(translator)

    text = re.sub('[ ]+',' ', text) # Remove double spaces

    return text.strip()

def get_tree(text):
    return text

def get_sentence_trees(tree):
    sents = tree.split('\n\n')
    sents = [sent.strip() for sent in sents]
    return [nltk.Tree.fromstring(sent) for sent in sents]

# Move this to parsers?
def get_wordcount(text):
    text_tree = get_tree(text)
    sents_trees = get_sentence_trees(text_tree)
    sents_tagged = [list(chain(*list(chain(*[[tree.pos() for tree in sents_tree]])))) for sents_tree in sents_trees]
    sents_words = [[item[0] for item in sent_tagged] for sent_tagged in sents_tagged]
    return sum(len(i) for i in sents_words)

def get_sentcount(text):
    text_tree = get_tree(text)
    sents_trees = get_sentence_trees(text_tree)
    return len(sents_trees)

### Normalized by wordcount

@textual_feature(tokenize_type=None)
def pronouns(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    return len(re.findall(r'\bPRO ', text)) / norm

@textual_feature(tokenize_type=None)
def determiners(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    return len(re.findall(r'\bD\^? ', text)) / norm

@textual_feature(tokenize_type=None)
def reflexives(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    return len(re.findall(r'\((.*?RFL|PRO\+.*?) ', text)) / norm

@textual_feature(tokenize_type=None)
def interrogatives(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    return len(re.findall(r'\(\. \?\)', text, re.IGNORECASE)) / norm

@textual_feature(tokenize_type=None)
def prepositions(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    return len(re.findall(r'\(P ', text)) / norm

@textual_feature(tokenize_type=None)
def superlatives(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    return len(re.findall(r'\bADJS ', text)) / norm

@textual_feature(tokenize_type=None)
def conjunct(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    return len(re.findall(r'\bCONJ ', text)) / norm

@textual_feature(tokenize_type=None)
def modals(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    return len(re.findall(r'\bMD ', text)) / norm

@textual_feature(tokenize_type=None)
def suma(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    cat = 'sum som somme summe some sume'.split()
    cat_re = '|'.join(cat)
    return len(re.findall(rf'\(.+ ({cat_re})\)', text, re.IGNORECASE)) / norm

@textual_feature(tokenize_type=None)
def ilca(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    cat = 'ilke ilche yche ilk ilce ilka ilca elke'.split()
    cat_re = '|'.join(cat)
    return len(re.findall(rf'\(.+ ({cat_re})\)', text, re.IGNORECASE)) / norm

@textual_feature(tokenize_type=None)
def othr(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    cat = ' o\+der o\+ter other o\+tere o\+dere othere'.split()
    cat_re = '|'.join(cat)
    return len(re.findall(rf'\(.+ ({cat_re})\)', text, re.IGNORECASE)) / norm

@textual_feature(tokenize_type=None)
def gif(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    cat = 'gif \+gif if ife iffe yf yif yiff yef \+giffe \+gyf \+gef'.split()
    cat_re = '|'.join(cat)
    return len(re.findall(rf'\(.+ ({cat_re})\)', text, re.IGNORECASE)) / norm

@textual_feature(tokenize_type=None)
def exclams(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    cat = 'o lo alas'.split()
    cat_re = '|'.join(cat)
    exclam = len(re.findall(rf'\(.+ ({cat_re})\)', text, re.IGNORECASE))
    exclam += len(re.findall(r'\bweilaw.+?\)', text, re.IGNORECASE))
    return exclam / norm

@textual_feature(tokenize_type=None)
def temporalcausal(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    cat = ['\+tonne', '\+tanne', '\+tane', '\+tenne', '\+tene', 'thonne', 'thanne', 'thane', 'thanne', 'thene',
           'whonne', 'whanne', 'whane', 'whenne', 'whene',
           'syne', 'synne', 'si\+t\+tenn', 'se\+ten', 'su\+t\+te',
           'lest', 'leste',
           'because', 'bycause', 'bicause', 'be cause', 'bi cause', 'by cause',
           'mid \+tam',
           'for\+di \+tat',
           'swa lang', 'so lang', 'sa lang', 'swa lange', 'so lange', 'sa lange',
           'swa \+tat', 'so \+tat', 'sa \+tat', 'swa that', 'so that', 'sa that']

    markers = 0
    text_tree = get_tree(text)
    sents_trees = get_sentence_trees(text_tree)
    sents_tagged = [list(chain(*list(chain(*[[tree.pos() for tree in sents_tree]])))) for sents_tree in sents_trees]
    sents_words = [[item[0] for item in sent_tagged] for sent_tagged in sents_tagged]
    plaintext = " ".join([" ".join(sent_word) for sent_word in sents_words])
    for item in cat:
        matches = re.findall(rf' {item}\b', plaintext, re.IGNORECASE)
        markers += len(matches)
    return markers / norm

### Normalized by sentcount

@textual_feature(tokenize_type=None)
def relclausesentences(text, normalize=False):
    norm = get_sentcount(text) if normalize else 1
    relclausesents = 0
    text_tree = get_tree(text)
    sents_trees = get_sentence_trees(text_tree)
    for sent_tree in sents_trees:
        rel_subtrees = list(sent_tree.subtrees(filter=lambda x: x.label().startswith('CP-REL') or x.label().startswith('CP-FRL') or x.label().startswith('CP-CAR')))
        if len(rel_subtrees) > 0:
            relclausesents += 1
    return relclausesents / norm

### Not normalized

# How to count spaces, punc, etc.?
@textual_feature(tokenize_type=None)
def meansent(text):
    text_tree = get_tree(text)
    sents_trees = get_sentence_trees(text_tree)
    sents_tagged = [list(chain(*list(chain(*[[tree.pos() for tree in sents_tree]])))) for sents_tree in sents_trees]
    sents_words = [[item[0] for item in sent_tagged if item[1] != 'ID' and item[1] != 'CODE'] for sent_tagged in sents_tagged]
    sents_text = [preprocess(" ".join(sent_word)) for sent_word in sents_words]
    sents_len = [len(sent) for sent in sents_text]
    return np.mean(sents_len)

@textual_feature(tokenize_type=None)
def avgrelclause(text):
    relclauselens = []
    text_tree = get_tree(text)
    sents_trees = get_sentence_trees(text_tree)
    rels = []
    rels_trees = []
    for sent_tree in sents_trees:
        rel_subtrees = sent_tree.subtrees(filter=lambda x: x.label().startswith('CP-REL') or x.label().startswith('CP-FRL') or x.label().startswith('CP-CAR'))
        rels_trees.append(rel_subtrees)
    for rel_tree in rels_trees:
        rel_tagged = [list(chain(*list(chain(*[[tree.pos() for tree in rel_tree]])))) for rel_tree in rels_trees]
        rel_tree_words = [[item[0] for item in rel_tagged] for rel_tagged in rel_tagged]
        for item in rel_tree_words:
            if item:
                rels.append(preprocess(" ".join(item)))
    relclauselens = [len(rel) for rel in rels]
    return np.mean(relclauselens)
