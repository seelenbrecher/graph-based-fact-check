import configparser
import json
import spacy
from spacy.matcher import Matcher
import sys
import timeit
from tqdm import tqdm
import numpy as np
blacklist = set(["-PRON-", "actually", "likely", "possibly", "want",
                 "make", "my", "someone", "sometimes_people", "sometimes","would", "want_to",
                 "one", "something", "sometimes", "everybody", "somebody", "could", "could_be"
                 ])


concept_vocab = set()
config = configparser.ConfigParser()
config.read("paths.cfg")
with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
    cpnet_vocab = [l.strip() for l in list(f.readlines())]
cpnet_vocab = [c.replace("_", " ") for c in cpnet_vocab]

## generate concept2id and id2concept
def load_concept_vocab():
    vocab = []
    with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
        vocab = [l.strip() for l in list(f.readlines())]
    concept2id = {}
    id2concept = {}
    for indice, cp in enumerate(vocab):
        concept2id[cp] = indice
        id2concept[indice] = cp
    return concept2id, id2concept

concept2id, id2concept = load_concept_vocab()


def lemmatize(nlp, concept):

    doc = nlp(concept.replace("_"," "))
    lcs = set()
    lcs.add("_".join([token.lemma_ for token in doc])) # all lemma
    return lcs

def load_matcher(nlp):
    config = configparser.ConfigParser()
    config.read("paths.cfg")
    with open(config["paths"]["matcher_patterns"], "r", encoding="utf8") as f:
        all_patterns = json.load(f)

    matcher = Matcher(nlp.vocab)
    for concept, pattern in tqdm(all_patterns.items(), desc="Adding patterns to Matcher."):
        matcher.add(concept, None, pattern)
    return matcher

def ground_mentioned_concepts(nlp, matcher, s, ans = ""):
    global concept2id
    s = s.lower()
    doc = nlp(s)
    matches = matcher(doc)

    mentioned_concepts = {}
    span_to_concepts = {}
    for match_id, start, end in matches:
        span = doc[start:end].text  # the matched span
        if len(set(span.split(" ")).intersection(set(ans.split(" ")))) > 0:
            continue
        original_concept = nlp.vocab.strings[match_id]
        # print("Matched '" + span + "' to the rule '" + string_id)

        if len(original_concept.split("_")) == 1:
            original_concept = list(lemmatize(nlp, original_concept))[0]

        if span not in span_to_concepts:
            span_to_concepts[span] = set()

        span_to_concepts[span].add(original_concept)

    for span, concepts in span_to_concepts.items():
        concepts_sorted = list(concepts)
        concepts_sorted.sort(key=len)

        # mentioned_concepts.update(concepts_sorted[0:2])

        shortest = concepts_sorted[0:3] #
        for c in shortest:
            if c in blacklist:
                continue
            lcs = lemmatize(nlp, c)
            intersect = lcs.intersection(shortest)
            if len(intersect)>0:
                c = list(intersect)[0]
                if c in concept2id:
                    mentioned_concepts[span] = c
                    break
            else:
                if c in concept2id:
                    mentioned_concepts[span] = c
                    break

    
    mentioned_concepts_with_indices = []
    for match_id, start, end in matches:
        span = doc[start:end].text
        if span in mentioned_concepts:
            concept = mentioned_concepts[span]
            concept_id = concept2id[concept]
            mentioned_concepts_with_indices.append([start, end, span, concept, concept_id])

    mentioned_concepts_with_indices = sorted(mentioned_concepts_with_indices, key=lambda x: (x[1],-x[0])) # sort based on end then start
    
    # mentioned_concepts_with_indice with filtered intersection
    res = []
    for mc in reversed(mentioned_concepts_with_indices):
        if len(res) == 0:
            res.append(mc)
        elif mc[1] <= res[-1][0]: # no intersection between current concept, and last included concepts 
            res.append(mc)
    
    res.reverse()
    
    return res

def hard_ground(nlp, sent):
    global cpnet_vocab
    sent = sent.lower()
    doc = nlp(sent)
    res = []
    for idx, t in enumerate(doc):
        if t.lemma_ in cpnet_vocab and t.lemma_ in concept2id:
            concept_id = concept2id[t.lemma_]
            res.append([idx, idx + 1, str(t), str(t.lemma_), concept_id])
    return res

def match_mentioned_concepts(nlp, matcher, sent):
    # print("Begin matching concepts.")
    all_concepts = ground_mentioned_concepts(nlp, matcher, sent)
    if len(all_concepts)==0:
        all_concepts = hard_ground(nlp, sent) # not very possible
        print('hard ground', sent)

    return all_concepts

def get_nlp_and_matcher():
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    matcher = load_matcher(nlp)
    return nlp, matcher

def test():
    nlp, matcher = get_nlp_and_matcher()
    res = match_mentioned_concepts(nlp, matcher, sent="Sometimes people say that someone stupid has no swimming pool.")
    print(res)