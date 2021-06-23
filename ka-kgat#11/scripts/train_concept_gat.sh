# python train.py --outdir ../checkpoint/kgat \
# --train_path ../data/bert_train.json \
# --valid_path ../data/bert_dev.json \
# --bert_pretrain ../bert_base \
# --postpretrain ../pretrain/save_model/model.best.pt

# use concept + gat
CUDA_VISIBLE_DEVICES=8 python train.py --outdir ../checkpoint/ka-kgat-concept-gat-1 \
--train_path ../data/fever_with_concepts/bert_train_concept.json \
--valid_path ../data/fever_with_concepts/bert_eval_concept.json \
--bert_pretrain ../bert_base \
--postpretrain ../checkpoint/pretrain/model.best.pt \
--use_concept \
--span_use_gat
