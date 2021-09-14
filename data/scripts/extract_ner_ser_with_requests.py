import requests

import argparse
import configparser
import json
import math
import multiprocessing
from tqdm import tqdm
import re
import time

def get_ner(sentence):
    headers = {
        "authority": "demo.allennlp.org",
        "sec-ch-ua-mobile": "?0",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
        "content-type": "text/plain;charset=UTF-8" ,
        "accept": "*/*" ,
        "origin": "https:/demo.allennlp.org" ,
        "sec-fetch-site": "same-origin" ,
        "sec-fetch-mode": "cors" ,
        "sec-fetch-dest": "empty" ,
        "referer": "https://demo.allennlp.org/named-entity-recognition/fine-grained-ner" ,
        "accept-language": "en-US,en;q=0.9" ,
        "cookie": "_ga=GA1.2.1131350624.1612546012; _gid=GA1.2.690578522.1621840252" ,
    }
    data = {'sentence': sentence}
    data = json.dumps(data)
    r = requests.post("https://demo.allennlp.org/api/fine-grained-ner/predict", headers=headers, data=data)
    res = json.loads(r.text)
    return res


def get_srl(sentence):
    headers = {
        "authority": "demo.allennlp.org",
        "sec-ch-ua-mobile": "?0",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
        "content-type": "text/plain;charset=UTF-8" ,
        "accept": "*/*" ,
        "origin": "https:/demo.allennlp.org" ,
        "sec-fetch-site": "same-origin" ,
        "sec-fetch-mode": "cors" ,
        "sec-fetch-dest": "empty" ,
        "referer": "https://demo.allennlp.org/semantic-role-labeling" ,
        "accept-language": "en-US,en;q=0.9" ,
        "cookie": "_ga=GA1.2.1131350624.1612546012; _gid=GA1.2.690578522.1621840252" ,
    }
    data = {'sentence': sentence}
    data = json.dumps(data)
    r = requests.post("https://demo.allennlp.org/api/semantic-role-labeling/predict", headers=headers, data=data)
    res = json.loads(r.text)
    return res





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
    
    print('read cur data, recovery')
    cur_data = {}
    try:
        with open('{}_{}.json'.format(args.output, batch), 'r') as f:
            for x in f:
                x = json.loads(x)
                cur_data[x['id']] = x
    except:
        print('cur data not found')

    print('process batch = {}/{}, start_id = {}, end_id = {}'.format(batch, n_batch, s_id, e_id))
    with open('{}_{}.json'.format(args.output, batch), 'w') as f_out:
        for instance in tqdm(input_data[s_id:e_id]):
            id = instance['id']
            if id in cur_data:
                f_out.write('{}\n'.format(json.dumps(cur_data[id])))
            else:
                time.sleep(1)
                res = {'id': id}
                claim = process_sent(instance['claim'])
                evis = [process_sent(a[2]) for a in instance['evidence']]

                res['claim_srl'] = get_srl(claim)
                ner = get_ner(claim)
                del ner['logits']
                res['claim_ner'] = ner

                res['evis_ner'] = []
                res['evis_srl'] = []
                for evi in evis:
                    ner = get_ner(evi)
                    del ner['logits']
                    res['evis_ner'].append(ner)
                    res['evis_srl'].append(get_srl(evi))

                f_out.write('{}\n'.format(json.dumps(res)))
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
