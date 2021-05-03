# python train.py --outdir ../checkpoint/kgat \
# --train_path ../data/bert_train.json \
# --valid_path ../data/bert_dev.json \
# --bert_pretrain ../bert_base \
# --postpretrain ../pretrain/save_model/model.best.pt

# CUDA_VISIBLE_DEVICES=12 python train.py --outdir ../checkpoint/chunked-kgat \
# --train_path ../data/standard_srl/srl_bert_train.json \
# --valid_path ../data/standard_srl/srl_bert_dev.json \
# --bert_pretrain ../bert_base \
# --postpretrain ../checkpoint/pretrain/model.best.pt \
# --infermodel_pretrain ../checkpoint/kgat/model.best.pt \
# --train_batch_size 1
# --max_len 60

# train attention block with KGAT as the backbone
# CUDA_VISIBLE_DEVICES=12 python train.py --outdir ../checkpoint/chunked-kgat-2 \
# --train_path ../data/standard_srl/srl_bert_train_complex.json \
# --valid_path ../data/standard_srl/srl_bert_dev.json \
# --bert_pretrain ../bert_base \
# --postpretrain ../checkpoint/pretrain/model.best.pt \
# --infermodel_pretrain ../checkpoint/kgat/model.best.pt \
# --train_batch_size 1 \
# --valid_batch_size 1 \
# --freeze_inference_model \
# --max_len 60

CUDA_VISIBLE_DEVICES=12 python train.py --outdir ../checkpoint/chunked-kgat-3 \
--train_path ../data/standard_srl/srl_bert_train_complex.json \
--valid_path ../data/standard_srl/srl_bert_dev_complex.json \
--bert_pretrain ../bert_base \
--postpretrain ../checkpoint/pretrain/model.best.pt \
--infermodel_pretrain ../checkpoint/kgat/model.best.pt \
--freeze_inference_model \
--num_train_epoch 50 \
--attn_hidden_size 1536
