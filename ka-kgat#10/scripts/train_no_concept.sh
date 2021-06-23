# python train.py --outdir ../checkpoint/kgat \
# --train_path ../data/bert_train.json \
# --valid_path ../data/bert_dev.json \
# --bert_pretrain ../bert_base \
# --postpretrain ../pretrain/save_model/model.best.pt

# no concept. but merge some sub-word to span level, based on mentioned concept
CUDA_VISIBLE_DEVICES=13 python train.py --outdir ../checkpoint/ka-kgat-no-concept \
--train_path ../data/fever_with_concepts/bert_train_concept.json \
--valid_path ../data/fever_with_concepts/bert_dev_concept.json \
--bert_pretrain ../bert_base \
--postpretrain ../checkpoint/pretrain/model.best.pt
