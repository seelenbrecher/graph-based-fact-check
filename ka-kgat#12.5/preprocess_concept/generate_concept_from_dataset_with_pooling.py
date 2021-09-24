import argparse
import configparser
import json
import math
import multiprocessing
from concept_extractor import get_nlp_and_matcher, match_mentioned_concepts

def worker(args, nlp, matcher, batch, n_batch):
    """thread worker function"""
    input_data = []
    with open(args.input, 'r') as f_in:
        for line in f_in:
            data = json.loads(line)
            input_data.append(data)

    n_data = len(input_data)
    item_cnt = math.ceil(n_data/n_batch)
    s_id = batch * item_cnt
    e_id = min((batch + 1) * item_cnt, len(input_data))
    print('process batch = {}/{}, start_id = {}, end_id = {}'.format(batch, n_batch, s_id, e_id))
    with open('{}_{}.json'.format(args.output, batch), 'w') as f_out:
        for data in input_data[s_id:e_id]:
            claim = data['claim']
            data['claim_concepts'] = match_mentioned_concepts(nlp, matcher, sent=claim)
            for evi_idx, evi in enumerate(data['evidence']):
                ev = evi[2]
                data['evidence'][evi_idx].append(match_mentioned_concepts(nlp, matcher, sent=ev))

            f_out.write('{}\n'.format(json.dumps(data)))
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', type=str)
    parser.add_argument('--output', dest='output', type=str)
    args = parser.parse_args()

    nlp, matcher = get_nlp_and_matcher()
    for batch in range(8):
        p = multiprocessing.Process(target=worker, args=(args, nlp, matcher, batch, 8,))
        p.start()


if __name__=='__main__':
    main()