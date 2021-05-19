# python train.py --outdir ../checkpoint/kgat \
# --train_path ../data/bert_train.json \
# --valid_path ../data/bert_dev.json \
# --bert_pretrain ../bert_base \
# --postpretrain ../pretrain/save_model/model.best.pt

CUDA_VISIBLE_DEVICES=5 python train.py --outdir ../checkpoint/reproduce-kgat-1 \
--train_path ../data/bert_train.json \
--valid_path ../data/bert_dev.json \
--bert_pretrain ../bert_base \
--postpretrain ../checkpoint/pretrain/model.best.pt
