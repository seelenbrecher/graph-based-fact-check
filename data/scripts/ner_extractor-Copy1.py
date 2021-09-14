import argparse
import configparser
import json
import math
import multiprocessing
from tqdm import tqdm
import re

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging



def process_sent(sentence):
    sentence = re.sub(" LSB.*?RSB", "", sentence)
    sentence = re.sub("LRB RRB ", "", sentence)
    sentence = re.sub("LRB", " ( ", sentence)
    sentence = re.sub("RRB", " )", sentence)
    sentence = re.sub("--", "-", sentence)
    sentence = re.sub("``", '"', sentence)
    sentence = re.sub("''", '"', sentence)

    return sentence

def process_wiki_title(title):
    title = re.sub("_", " ", title)
    title = re.sub("LRB", " ( ", title)
    title = re.sub("RRB", " )", title)
    title = re.sub("COLON", ":", title)
    return title

def worker(args, batch, n_batch):
    print('load_model', batch)
    srl_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
    ner_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/fine-grained-ner.2021-02-11.tar.gz")
    print('end load model', batch)
    """thread worker function"""
    input_data = []
    print('load data', batch)
    with open(args.input, 'r') as f_in:
        for line in f_in:
            data = json.loads(line)
            input_data.append(data)
    print('end load data', batch)

    n_data = len(input_data)
    item_cnt = math.ceil(n_data/n_batch)
    s_id = batch * item_cnt
    e_id = min((batch + 1) * item_cnt, len(input_data))
    print('process batch = {}/{}, start_id = {}, end_id = {}'.format(batch, n_batch, s_id, e_id))
    with open('{}_{}.json'.format(args.output, batch), 'w') as f_out:
        for instance in tqdm(input_data[s_id:e_id]):
            res = {}
            claim = process_sent(instance['claim'])
            evis = [process_sent(a[2]) for a in instance['evidence']]

            res['claim_srl'] = srl_predictor.predict(claim)
            ner = ner_predictor.predict(claim)
            del ner['logits']
            res['claim_ner'] = ner

            res['evis_ner'] = []
            res['evis_srl'] = []
            for evi in evis:
                ner = ner_predictor.predict(evi)
                del ner['logits']
                res['evis_ner'].append(ner)
                res['evis_srl'].append(srl_predictor.predict(evi))
            
            f_out.write('{}\n'.format(json.dumps(data)))
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', type=str)
    parser.add_argument('--output', dest='output', type=str)
    args = parser.parse_args()

    for batch in range(8):
        p = multiprocessing.Process(target=worker, args=(args, batch, 8,))
        p.start()


if __name__=='__main__':
    main()