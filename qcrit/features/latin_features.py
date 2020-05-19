# pylint: disable = missing-docstring
'''
Latin features
'''
import re

from functools import reduce

from qcrit.textual_feature import textual_feature

#---------------------------------------------------------------------------
# Constants

RELATIVES = {'qui', 'cuius', 'cui', 'quem', 'quo', 'quae', 'quam',
            'qua', 'quod', 'quorum', 'quibus', 'quos', 'quarum', 'quas'}

ABBREVIATIONS = {"Caes",  "Iul",  "Plur",  "Aug",  "Prid",  "Id",  "Kal",  "Kl",  "Non",  "Med",  "Ex",  "In",  "Ian",  "Feb",  "Febr",  "Mart",  "Apr",  "Iun",  "Mai",  "Nov",  "Oct",  "Sept",  "Dec",  "Fin",  "Cos",  "Coss",  "Pr",  "Sext",  "Ann",  "Sal",  "Imp",  "Tr",  "Pl",  "Leg",  "Aed",  "Cens",  "Agr",  "Ant",  "Aur",  "Cn",  "Scrib",  "Fab",  "Pom",  "Quir",  "Pup",  "An",  "Ter",  "Op",  "Germ",  "Gn",  "Ap",  "M’",  "Mai",  "Mam",  "Men",  "Min",  "Oct",  "Opet",  "Pro",  "Quint",  "Quinct",  "Sec",  "Ser",  "Sert",  "Serv",  "Seq",  "Sex",  "Sp",  "St",  "Ti",  "Tib",  "Vol",  "Vop",  "Aem",  "Aim",  "Rom",  "Pont",  "Imp",  "Max",  "Coll",  "Ob ",  "Lib",  "Cir",  "Ver",  "III",  "Eq",  "eq",  "I",  "II",  "III",  "IIII",  "IV",  "V",  "VI",  "VII",  "VIII",  "VIIII",  "IX",  "X",  "XI",  "XII",  "XIII",  "XIIII",  "XIV",  "XV",  "XVI",  "XVII",  "XVIII",  "XVIIII",  "XIX",  "XX",  "XXI",  "XXII",  "XXIII",  "XXIIII",  "XXIV",  "XXV",  "XXVI",  "XXVII",  "XXVIII",  "XXVIIII",  "XXIX",  "XXX",  "XXXI",  "XXXII",  "XXXIII",  "XXXIIII",  "XXXIV",  "XXXV",  "XXXVI",  "XXXVII",  "XXXVIII",  "XXXVIIII",  "XXXIX",  "XXXX",  "XL",  "XLI",  "XLII",  "XLIII",  "XLIIII",  "XLIV",  "XLV",  "XLVI",  "XLVII",  "XLVIII",  "XLVIIII",  "XLIX",  "L",  "LI",  "LII",  "LIII",  "LIIII",  "LIV",  "LV",  "LVI",  "LVII",  "LVIII",  "LVIIII",  "LIX",  "LX",  "LXI",  "LXII",  "LXIII",  "LXIIII",  "LXIV",  "LXV",  "LXVI",  "LXVII",  "LXVIII",  "LXVIIII",  "LXIX",  "LXX",  "LXXI",  "LXXII",  "LXXIII",  "LXXIIII",  "LXXIV",  "LXXV",  "LXXVI",  "LXXVII",  "LXXVIII",  "LXXVIIII",  "LXXIX",  "LXXX",  "LXXXI",  "LXXXII",  "LXXXIII",  "LXXXIIII",  "LXXXIV",  "LXXXV",  "LXXXVI",  "LXXXVII",  "LXXXVIII",  "LXXXVIIII",  "LXXXIX",  "LXXXX",  "XC",  "XCI",  "XCII",  "XCIII",  "XCIIII",  "XCIV",  "XCV",  "XCVI",  "XCVII",  "XCVIII",  "XCVIIII",  "XCIX",  "C",  "CC",  "CCC",  "CCCC",  "CD",  "D",  "DC",  "DCC",  "DCCC",  "DCCCC",  "CM",  "M",  "MM",  "MMM",  "MMMM"}

#---------------------------------------------------------------------------
# Helpers

def _count_target_freq(list_of_words, targets):
    num_target = 0
    num_characters = 0
    for word in list_of_words:
        num_target += 1 if word in targets else 0
        num_characters += len(word)
    return num_target / len(list_of_words)

def _count_target_bigram_freq(list_of_words, monograms, bigrams):
    # If monograms and bigrams contained any of the same elements, an ambiguity would arise
    assert all(ele not in bigrams for ele in monograms)
    assert None not in monograms and None not in bigrams

    num_target = 0
    num_characters = 0
    bigram_first_half = None
    for word in list_of_words:
        if word in monograms:
            num_target += 1
            bigram_first_half = None
        elif word in bigrams:
            bigram_first_half = word
        elif bigram_first_half in bigrams and word in bigrams[bigram_first_half]:
            num_target += 1
            bigram_first_half = None
        else:
            bigram_first_half = None
        num_characters += len(word)
    return num_target / len(list_of_words)

# TERMINAL_PUNCTUATION = ('.', '?', '!')
PUNC_STRIP = (',', ';', ':')
# setup_tokenizers(terminal_punctuation=TERMINAL_PUNCTUATION)

#---------------------------------------------------------------------------
# List 0
# https://github.com/qcrit/DSH-2018-LatinProseVerse/blob/master/stylometry/imports/api/stylometry.js#L999-L1014

@textual_feature(tokenize_type='words')
def personal_pronoun(text):
    text = [t for t in text if t not in PUNC_STRIP]
    return _count_target_freq(
        text,
        {
        'ego', 'mei', 'mihi', 'me', 'tu', 'tui', 'tibi', 'te', 'nos', 'nostri',
        'nobis', 'vos', 'vestri', 'vobis', 'uos', 'uestri', 'uobis', 'mi', 'nostrum',
        'vestrum', 'vostrum', 'vostri', 'uestrum', 'uostrum', 'uostri', 'egoque', 'meique',
        'mihique', 'meque', 'tuque', 'tuique', 'tibique', 'teque', 'nosque', 'nostrique',
        'nobisque', 'vosque', 'vestrique', 'vobisque', 'uosque', 'uestrique', 'uobisque',
        'mique', 'nostrumque', 'vestrumque', 'vostrumque', 'vostrique', 'uestrumque',
        'uostrumque', 'uostrique'
        }
    )

@textual_feature(tokenize_type='words')
def demonstrative_pronoun(text):
    text = [t for t in text if t not in PUNC_STRIP]
    return _count_target_freq(
        text,
        {
            'hic', 'hunc', 'huius', 'huic', 'hoc', 'haec', 'hanc', 'hac', 'hi',
            'hos', 'horum', 'his', 'hae', 'has', 'harum', 'ho', 'ha', 'ille',
            'illum', 'illius', 'illi', 'illo', 'illa', 'illam', 'illud', 'illos',
            'illorum', 'illis', 'illas', 'illarum', 'illae', 'is', 'eum', 'eius',
            'ei', 'eo', 'ea', 'eam', 'id', 'ii', 'eos', 'eorum', 'eis', 'iis',
            'eae', 'eas', 'earum', 'illic', 'illaec', 'illuc', 'illic', 'illoc',
            'illunc', 'illanc', 'illac', 'hicque', 'huncque', 'huiusque', 'huicque',
            'hocque', 'haecque', 'hancque', 'hacque', 'hique', 'hosque', 'horumque',
            'hisque', 'haeque', 'hasque', 'harumque', 'hoque', 'haque', 'illeque',
            'illumque', 'illiusque', 'illique', 'illoque', 'illaque', 'illamque',
            'illudque', 'illosque', 'illorumque', 'illisque', 'illasque', 'illarumque',
            'illaeque', 'isque', 'eumque', 'eiusque', 'eique', 'eoque', 'eaque',
            'eamque', 'idque', 'iique', 'eosque', 'eorumque', 'eisque', 'iisque',
            'eaeque', 'easque', 'earumque', 'illicque', 'illaecque', 'illucque',
            'illicque', 'illocque', 'illuncque', 'illancque', 'illacque'
        }
    )

@textual_feature(tokenize_type='words')
def quidam(text):
    text = [t for t in text if t not in PUNC_STRIP]
    return _count_target_freq(
        text,
        {
            'quidam', 'quendam', 'cuiusdam', 'cuidam', 'quodam', 'quaedam',
            'quandam', 'quodam', 'quoddam', 'quosdam', 'quorundam', 'quibusdam',
            'quasdam', 'quarundam', 'quiddam', 'quoddam', 'quadam', 'quidamque',
            'quendamque', 'cuiusdamque', 'cuidamque', 'quodamque', 'quaedamque',
            'quandamque', 'quodamque', 'quoddamque', 'quosdamque', 'quorundamque',
            'quibusdamque', 'quasdamque', 'quarundamque', 'quiddamque', 'quoddamque',
            'quadamque'
        }
    )

@textual_feature(tokenize_type='words')
def reflexive_pronoun(text):
    text = [t for t in text if t not in PUNC_STRIP]
    return _count_target_freq(
        text,
        {
            'se', 'sibi', 'sese', 'sui', 'seque', 'sibique', 'seseque', 'suique'
        }
    )

@textual_feature(tokenize_type='words')
def iste(text):
    text = [t for t in text if t not in PUNC_STRIP]
    return _count_target_freq(
        text,
        {
            'iste', 'istum', 'istius', 'isti', 'isto', 'ista', 'istam', 'istud',
            'istos', 'istorum', 'istis', 'istas', 'istarum', 'iste', 'istum',
            'istius', 'isti', 'isto', 'ista', 'istam', 'istud', 'istos',
            'istorum', 'istis', 'istas', 'istarum', 'isteque', 'istumque',
            'istiusque', 'istique', 'istoque', 'istaque', 'istamque', 'istudque',
            'istosque', 'istorumque', 'istisque', 'istasque', 'istarumque'
        }
    )

@textual_feature(tokenize_type='words')
def alius(text):
    text = [t for t in text if t not in PUNC_STRIP]
    return _count_target_freq(
        text,
        {
            'alius', 'alium', 'alii', 'alio', 'alia', 'aliam', 'aliud', 'alios',
            'aliorum', 'aliis', 'aliae', 'alias', 'aliarum', 'aliusque', 'aliumque',
            'aliique', 'alioque', 'aliaque', 'aliamque', 'aliudque', 'aliosque',
            'aliorumque', 'aliisque', 'aliaeque', 'aliasque', 'aliarumque'
        }
    )

@textual_feature(tokenize_type='words')
def ipse(text):
    text = [t for t in text if t not in PUNC_STRIP]
    return _count_target_freq(
        text,
        {
            'ipse', 'ipsum', 'ipsius', 'ipsi', 'ipso', 'ipsa', 'ipsam', 'ipsos',
            'ipsorum', 'ipsas', 'ipsarum', 'ipseque', 'ipsumque', 'ipsiusque',
            'ipsique', 'ipsoque', 'ipsaque', 'ipsamque', 'ipsosque', 'ipsorumque',
            'ipsasque', 'ipsarumque'
        }
    )

@textual_feature(tokenize_type='words')
def idem(text):
    text = [t for t in text if t not in PUNC_STRIP]
    return _count_target_freq(
        text,
        {
            'idem', 'eundem', 'eiusdem', 'eidem', 'eodem', 'eadem', 'eandem',
            'iidem', 'eosdem', 'eorundem', 'eisdem', 'iisdem', 'eaedem', 'eedem',
            'easdem', 'earumdem', 'isdem', 'isdemque', 'idemque', 'eundemque',
            'eiusdemque', 'eidemque', 'eodemque', 'eademque', 'eandemque', 'iidemque',
            'eosdemque', 'eorundemque', 'eisdemque', 'iisdemque', 'eaedemque',
            'easdemque', 'earundemque'
        }
    )

@textual_feature(tokenize_type='words')
def priu(text):
    text = [t for t in text if t not in PUNC_STRIP]
    return _count_target_bigram_freq(text, {'priusquam'}, {'prius': {'quam'}})

@textual_feature(tokenize_type='words')
def anteq(text):
    text = [t for t in text if t not in PUNC_STRIP]
    return _count_target_bigram_freq(text, {'antequam',}, {'ante': {'quam'}})

@textual_feature(tokenize_type='words')
def quom(text):
    text = [t for t in text if t not in PUNC_STRIP]
    return _count_target_bigram_freq(text, {'quominus',}, {'quo': {'minus'}})

@textual_feature(tokenize_type='words')
def dum(text):
    text = [t for t in text if t not in PUNC_STRIP]
    return _count_target_freq(text, {'dum', 'dumque'})

@textual_feature(tokenize_type='words')
def quin(text):
    text = [t for t in text if t not in PUNC_STRIP]
    return _count_target_freq(text, {'quin',})

@textual_feature(tokenize_type='words')
def ut(text): # pylint: disable = invalid-name
    text = [t for t in text if t not in PUNC_STRIP]
    return _count_target_freq(text, {'ut', 'utei', 'utque'})

@textual_feature(tokenize_type='words')
def conditional_clauses(text):
    text = [t for t in text if t not in PUNC_STRIP]
    return _count_target_freq(text, {'si', 'nisi', 'quodsi', 'sin', 'dummodo'})

@textual_feature(tokenize_type='words')
def prepositions(text):
    text = [t for t in text if t not in PUNC_STRIP]
    return _count_target_freq(
        text,
        {
            'ab', 'abs', 'e', 'ex', 'apud', 'de', 'cis', 'erga', 'inter', 'ob',
            'penes', 'per', 'praeter', 'propter', 'trans', 'absque', 'pro',
            'tenus', 'sub', 'aque', 'abque', 'eque', 'exque', 'apudque', 'deque',
            'cisque', 'ergaque', 'interque', 'obque', 'penesque', 'perque',
            'praeterque', 'propterque', 'transque', 'proque', 'tenusque', 'subque'
        }
    )

#---------------------------------------------------------------------------
# List F
# https://github.com/qcrit/DSH-2018-LatinProseVerse/blob/master/stylometry/imports/api/stylometry.js#L1018-L1023

@textual_feature(tokenize_type='words')
def interrogative(text):
    text = [t for t in text if t not in PUNC_STRIP]
    num_target = 0
    num_characters = 0
    for word in text:
        num_target += 1 if '?' in word else 0
        num_characters += len(word)
    return num_target / len(text)

@textual_feature(tokenize_type='words')
def superlatives(text):
    text = [t for t in text if t not in PUNC_STRIP]
    num_superlatives = 0
    num_characters = 0
    for word in text:
        num_superlatives += 1 if 'issim' in word else 0
        num_characters += len(word)
    return num_superlatives / len(text)

@textual_feature(tokenize_type='words')
def startf(text):
    text = [t for t in text if t not in PUNC_STRIP]
    num_target = 0
    num_characters = 0
    consonants = (
        'b', 'c', 'd', 'f', 'g', 'j', 'k', 'l', 'm', 'n',
        'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z'
    )
    for i in range(len(text) - 1):
        num_target += 1 if text[i] == 'atque' and text[i + 1].startswith(consonants) else 0
        num_characters += len(text[i])
    num_characters += len(text[-1])
    return num_target / len(text)

@textual_feature(tokenize_type='words')
def end_gerund(text):
    text = [t for t in text if t not in PUNC_STRIP]
    num_gerund = 0
    num_characters = 0
    gerund_endings = ('ndum', 'ndus', 'ndorum', 'ndarum', 'ndumque', 'ndusque', 'ndorumque', 'ndarumque')
    for word in text:
        if word != 'nondum' and word.endswith(gerund_endings):
            num_gerund += 1
        num_characters += len(word)
    return num_gerund / len(text)

@textual_feature(tokenize_type='words')
def cum_clause(text):
    text = [t for t in text if t not in PUNC_STRIP]
    num_cum_clause = 0
    num_characters = 0
    cum_clause_endings = ('a', 'e', 'i', 'o', 'u', 'is', 'ibus', 'ebus', 'obus', 'ubus')
    for i in range(len(text) - 1):
        num_cum_clause += 1 if text[i] == 'cum' and not text[i + 1].endswith(cum_clause_endings) else 0
        num_characters += len(text[i])
    num_characters += len(text[-1])
    return num_cum_clause / len(text)

#---------------------------------------------------------------------------
# List C
# https://github.com/qcrit/DSH-2018-LatinProseVerse/blob/master/stylometry/imports/api/stylometry.js#L1027-L1031

@textual_feature(tokenize_type='words')
def multi_occurence(text):
    text = [t for t in text if t not in PUNC_STRIP]
    num_target = 0
    num_characters = 0
    conjunctions = {
        'et', 'atque', 'ac', 'aut', 'vel', 'uel', 'at', 'autem', 'sed', 'postquam',
        'ast', 'donec', 'dum', 'dummodo', 'enim', 'etiam', 'etiamtum', 'etiamtunc', 'etenim',
        'veruntamen', 'ueruntamen', 'uerumtamen', 'verumtamen', 'utrumnam', 'set', 'quocirca',
        'quia', 'quamquam', 'quanquam', 'nam', 'namque', 'nanque', 'nempe', 'dumque',
        'etiamque', 'quiaque'
    }
    for word in text:
        num_target += 1 if word in conjunctions or word.endswith('que') else 0
        num_characters += len(word)
    return num_target / len(text)

@textual_feature(tokenize_type='words')
def endf(text):
    text = [t for t in text if t not in PUNC_STRIP]
    num_target = 0
    num_characters = 0
    vocatives = ('e', 'i', 'a', 'u', 'ae', 'es', 'um', 'us')
    for i in range(len(text) - 1):
        num_target += 1 if text[i] == 'o' and text[i + 1].endswith(vocatives) else 0
        num_characters += len(text[i])
    num_characters += len(text[-1])
    return num_target / len(text)

#---------------------------------------------------------------------------
# Miscellaneous -- SORT!

@textual_feature(tokenize_type='sentence_words')
def mean_sentence(text):
    # fix
    len_of_all_words = reduce(
        lambda cur_len, line: cur_len + reduce(lambda word_len, word: word_len + len(word), line, 0),
        text,
        0
    )
    return len_of_all_words / len(text)


@textual_feature(tokenize_type='sentence_words')
def mean_relative(text):
    import re
    import statistics
    abbrev = [ABBREVIATION.lower() for ABBREVIATION in ABBREVIATIONS]
    sents = text
    sents = [[w for w in s if w.lower() not in abbrev] for s in sents]
    text_ = " ".join([" ".join(sent) for sent in sents])
    relative_group = "|".join(RELATIVES)
    pattern = rf'\b({relative_group})\b.+?[,:\.!;]'
    matches = re.finditer(pattern, text_)
    matches_ = []
    for match in matches:
        item = text_[match.start(0):match.end(0)]
        if "?" not in item:
            matches_.append(item)

    len_rel_clauses = [len(match) for match in matches_]
    if len_rel_clauses:
        return statistics.mean(len_rel_clauses)
    else:
        return 0

@textual_feature(tokenize_type='sentence_words')
def relatives(text):
    import re
    import statistics
    # abbrev = [ABBREVIATION.lower() for ABBREVIATION in ABBREVIATIONS]
    sents = [" ".join(words) for words in text]
    # sents = [[w for w in s if w.lower() not in abbrev] for s in sents]
    # text_ = " ".join([" ".join(sent) for sent in sents])
    relative_group = "|".join(RELATIVES)
    pattern = rf'\b({relative_group})\b.+?[,:\.!;]'
    rel_count = 0
    for sent in sents:
        matches = re.finditer(pattern, sent)
        matches_ = []
        for match in matches:
            item = sent[match.start(0):match.end(0)]
            if "?" not in item:
                matches_.append(item)
        if matches_:
            rel_count += 1
    return rel_count/len(sents)
