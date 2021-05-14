# pylint: disable = missing-docstring
'''
Middle English features
'''

from ..textual_feature import textual_feature
import re
import nltk
from itertools import chain
import numpy as np

def get_tree(text):
    return re.sub(r'\n\n[\s\S]*?\n\n', '\n\n', text).strip()

def get_sentence_trees(tree):
    return [nltk.Tree.fromstring(tree_) for tree_ in tree.split('\n\n')]

# Move this to parsers?
def get_wordcount(text):
    text_tree = get_tree(text)
    sents_trees = get_sentence_trees(text_tree)
    sents_tagged = [list(chain(*list(chain(*[[tree.pos() for tree in sents_tree]])))) for sents_tree in sents_trees]
    sents_tagged = [[item for item in sent_tagged if item[1] != 'CD'] for sent_tagged in sents_tagged]
    sents_words = [[item[0] for item in sent_tagged] for sent_tagged in sents_tagged]
    sents_words = [[item.replace("'",'') for item in sent_word] for sent_word in sents_words] # Control for apostrophe
    sents_words = [[item for item in sent_word if item.isalpha() and (len(item) == 1 or not item.isupper())] for sent_word in sents_words]  # Remove tokens with punctuation
    sents_words = [sent_word for sent_word in sents_words if len(sent_word) > 2 and ''.join(sent_word) != 'No'] # Fix parser issue with N o and roman numerals
    return sum(len(i) for i in sents_words)

def get_sentcount(text):
    text_tree = get_tree(text)
    sents_trees = get_sentence_trees(text_tree)
    return len(sents_trees)

@textual_feature(tokenize_type=None)
def wordcount(text):
    return get_wordcount(text)

@textual_feature(tokenize_type=None)
def sentcount(text):
    return get_sentcount(text)

@textual_feature(tokenize_type=None)
def pronouns(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    return len(re.findall(r'\bPRP ', text)) / norm

@textual_feature(tokenize_type=None)
def determiners(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    return len(re.findall(r'\bDT\b', text)) / norm

@textual_feature(tokenize_type=None)
def some(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    return len(re.findall(r'\(DT some\)', text, re.IGNORECASE)) / norm

@textual_feature(tokenize_type=None)
def reflexives(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    return len(re.findall(r'\(PRP \w+sel(f|ves)\)', text, re.IGNORECASE)) / norm

@textual_feature(tokenize_type=None)
def ilk(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    return len(re.findall(r'\(DT the\) \(JJ same\)', text, re.IGNORECASE)) / norm

@textual_feature(tokenize_type=None)
def othr(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    return len(re.findall(r'\(JJ other\)', text, re.IGNORECASE)) / norm

@textual_feature(tokenize_type=None)
def conjunct(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    cat = ['and', 'both', 'bothe', 'boyth', 'but', 'butt', 'either', 'eyther', 'nor', 'nother', 'neither', 'neyther', 'ne', 'or', 'ore', '&']
    cat_re = '|'.join(cat)
    return len(re.findall(rf'\(.+ ({cat_re})\)', text, re.IGNORECASE)) / norm

@textual_feature(tokenize_type=None)
def relclausesentences(text, normalize=False):
    norm = get_sentcount(text) if normalize else 1
    relclausesents = 0
    text_tree = get_tree(text)
    sents_trees = get_sentence_trees(text_tree)
    for sent_tree in sents_trees:
        sent_sbar = [sbar for sbar in sent_tree.subtrees(filter=lambda x: x.label() == 'SBAR')]
        for sbar in sent_sbar:
            sbar_trees = sbar.subtrees()
            next(sbar_trees)
            if next(sbar_trees).label() == 'WHNP':
                relclausesents += 1
                break
    return relclausesents / norm

@textual_feature(tokenize_type=None)
def avgrelclause(text):
    relclauselens = []
    text_tree = get_tree(text)
    sents_trees = get_sentence_trees(text_tree)
    for sent_tree in sents_trees:
        sent_sbar = [sbar for sbar in sent_tree.subtrees(filter=lambda x: x.label() == 'SBAR')]
        for sbar in sent_sbar:
            sbar_trees = sbar.subtrees()
            next(sbar_trees)
            sbar_subtree = next(sbar_trees)
            if sbar_subtree.label() == 'WHNP':
                whnp_tagged = list(chain(*list(chain(*[[tree.pos() for tree in sbar]]))))
                whnp_words = [item[0] for item in whnp_tagged]
                whnp_text = ''.join(whnp_words)
                relclauselens.append(len(whnp_text))
    return np.mean(relclauselens)

@textual_feature(tokenize_type=None)
def conditional(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    return len(re.findall(r'\(IN if\)', text, re.IGNORECASE)) / norm

@textual_feature(tokenize_type=None)
def temporalcausal(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    cat = ['since', 'sens', 'sithence', 'syns', 'then', 'thenne', 'finally', 'although', 'althoughe', 'despite', 'because', 'consequently', 'therefore', 'thus', 'lest', 'leste', 'when', 'whenne']
    cat_re = '|'.join(cat)
    return len(re.findall(rf'\(.+ ({cat_re})\)', text, re.IGNORECASE)) / norm

# How to count spaces, punc, etc.?
@textual_feature(tokenize_type=None)
def meansent(text):
    text_tree = get_tree(text)
    sents_trees = get_sentence_trees(text_tree)
    sents_tagged = [list(chain(*list(chain(*[[tree.pos() for tree in sents_tree]])))) for sents_tree in sents_trees]
    sents_words = [[item[0] for item in sent_tagged] for sent_tagged in sents_tagged]
    sents_text = ["".join(sent_word) for sent_word in sents_words]
    sents_len = [len(sent) for sent in sents_text]
    return np.mean(sents_len)

@textual_feature(tokenize_type=None)
def interrogatives(text, normalize=False):
    norm = get_sentcount(text) if normalize else 1
    return len(re.findall(r'\(\. \?\)', text, re.IGNORECASE)) / norm

@textual_feature(tokenize_type=None)
def prepositions(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    return len(re.findall(r'\(PP \(IN ', text)) / norm

@textual_feature(tokenize_type=None)
def superlatives(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    return len(re.findall(r'\(JJS ', text)) / norm

@textual_feature(tokenize_type=None)
def exclams(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    cat = ['o', 'lo', 'alas']
    cat_re = '|'.join(cat)
    return len(re.findall(rf'\(.+ ({cat_re})\)', text, re.IGNORECASE)) / norm

@textual_feature(tokenize_type=None)
def modals(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    return len(re.findall(r'\bMD ', text)) / norm

# @textual_feature(tokenize_type=None)
def discoursemarkers(text, normalize=False):
    norm = get_wordcount(text) if normalize else 1
    cat = ['moreover', 'overall', 'in conclusion', 'at the end of the day', 'to the extent that', 'on the other hand', 'first of all', 'in the end']
    discmarks = 0
    text_tree = get_tree(text)
    sents_trees = get_sentence_trees(text_tree)
    sents_tagged = [list(chain(*list(chain(*[[tree.pos() for tree in sents_tree]])))) for sents_tree in sents_trees]
    sents_words = [[item[0] for item in sent_tagged] for sent_tagged in sents_tagged]
    plaintext = " ".join([" ".join(sent_word) for sent_word in sents_words])
    for item in cat:
        discmarks += len(re.findall(rf'\b{item}\b', plaintext, re.IGNORECASE))
    return discmarks / norm
