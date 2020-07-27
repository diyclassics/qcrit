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

@textual_feature(tokenize_type=None)
def pronouns(text):
    return len(re.findall(r'\bPRP ', text))

@textual_feature(tokenize_type=None)
def determiners(text):
    return len(re.findall(r'\bDT\b', text))

@textual_feature(tokenize_type=None)
def suma(text):
    return len(re.findall(r'\(DT some\)', text, re.IGNORECASE))

@textual_feature(tokenize_type=None)
def reflexives(text):
    return len(re.findall(r'\(PRP \w+sel(f|ves)\)', text))

@textual_feature(tokenize_type=None)
def ilca(text):
    return len(re.findall(r'\(DT the\) \(JJ same\)', text, re.IGNORECASE))

@textual_feature(tokenize_type=None)
def othr(text):
    return len(re.findall(r'\(JJ other\)', text, re.IGNORECASE))

@textual_feature(tokenize_type=None)
def conjunct(text):
    cat = ['and', 'both', 'bothe', 'boyth', 'but', 'butt', 'either', 'eyther', 'nor', 'nother', 'neither', 'neyther', 'ne', 'or', 'ore', '&']
    cat_re = '|'.join(cat)
    return len(re.findall(rf'\(.+ ({cat_re})\)', text, re.IGNORECASE))

@textual_feature(tokenize_type=None)
def relclausesentences(text):
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
    return relclausesents

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
def gif(text):
    return len(re.findall(r'\(IN if\)', text, re.IGNORECASE))

@textual_feature(tokenize_type=None)
def interrogatives(text):
    return len(re.findall(r'\(\. \?\)', text, re.IGNORECASE))

@textual_feature(tokenize_type=None)
def prepositions(text):
    return len(re.findall(r'\(PP \(IN ', text))

@textual_feature(tokenize_type=None)
def superlatives(text):
    return len(re.findall(r'\(JJS ', text))

@textual_feature(tokenize_type=None)
def exclams(text):
    cat = ['o', 'lo', 'alas']
    cat_re = '|'.join(cat)
    return len(re.findall(rf'\(.+ ({cat_re})\)', text, re.IGNORECASE))

@textual_feature(tokenize_type=None)
def modals(text):
    return len(re.findall(r'\bMD ', text))

@textual_feature(tokenize_type=None)
def discoursemarkers(text):
    cat = ['moreover', 'overall', 'in conclusion', 'at the end of the day', 'to the extent that', 'on the other hand', 'first of all', 'in the end']
    discmarks = 0
    text_tree = get_tree(text)
    sents_trees = get_sentence_trees(text_tree)
    sents_tagged = [list(chain(*list(chain(*[[tree.pos() for tree in sents_tree]])))) for sents_tree in sents_trees]
    sents_words = [[item[0] for item in sent_tagged] for sent_tagged in sents_tagged]
    plaintext = " ".join([" ".join(sent_word) for sent_word in sents_words])
    for item in cat:
        discmarks += len(re.findall(rf'\b{item}\b', plaintext, re.IGNORECASE))
    return discmarks
