import argparse
import configparser
import json
from concept_extractor import get_nlp_and_matcher, match_mentioned_concepts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', type=str)
    parser.add_argument('--output', dest='output', type=str)
    args = parser.parse_args()
    
    input_data = []
    with open(args.input, 'r') as f_in:
        for line in f_in:
            data = json.loads(line)
            input_data.append(data)
   
    nlp, matcher = get_nlp_and_matcher()
    with open(args.output, 'w') as f_out:
        for data in input_data:
            claim = data['claim']
            data['claim_concepts'] = match_mentioned_concepts(nlp, matcher, sent=claim)
            for evi_idx, evi in enumerate(data['evidence']):
                ev = evi[2]
                data['evidence'][evi_idx].append(match_mentioned_concepts(nlp, matcher, sent=ev))

            f_out.write('{}\n'.format(json.dumps(data)))


if __name__=='__main__':
    main()