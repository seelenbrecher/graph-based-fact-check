# CUDA_VISIBLE_DEVICES=8 python test.py --outdir ./output/ \
# --test_path ../data/standard_srl/srl_bert_eval.json \
# --bert_pretrain ../bert_base \
# --checkpoint ../checkpoint/chunked-kgat/model.best.pt \
# --name chunked-dev-1.json \
# --batch_size 1 \
# --max_len 60


# CUDA_VISIBLE_DEVICES=8 python test.py --outdir ./output/ \
# --test_path ../data/standard_srl/srl_bert_test.json \
# --bert_pretrain ../bert_base \
# --checkpoint ../checkpoint/chunked-kgat/model.best.pt \
# --name chunked-test-1.json \
# --batch_size 1 \
# --max_len 60

# naive-kgat
# CUDA_VISIBLE_DEVICES=9 python test.py --outdir ./output/ \
# --test_path ../data/standard_srl/srl_bert_eval.json \
# --bert_pretrain ../bert_base \
# --checkpoint ../checkpoint/reproduce-kgat/model.best.pt \
# --name naive-dev.json \
# --chunked_model naive \
# --batch_size 1 \
# --max_len 6

# for chunked-kgat-train-att with 768 att hidden size
# CUDA_VISIBLE_DEVICES=8 python test.py --outdir ./output/ \
# --test_path ../data/standard_srl/srl_bert_eval.json \
# --bert_pretrain ../bert_base \
# --checkpoint ../checkpoint/chunked-kgat-train-att/model.best.pt \
# --name chunked-dev-train-att.json \
# --batch_size 1

# for chunked-kgat-train-att-bigger with 1536 att hidden size
# CUDA_VISIBLE_DEVICES=12 python test.py --outdir ./output/ \
# --test_path ../data/standard_srl/srl_bert_eval.json \
# --bert_pretrain ../bert_base \
# --checkpoint ../checkpoint/chunked-kgat-train-att-bigger/model.best.pt \
# --name chunked-dev-train-att-bigger.json \
# --attn_hidden_size 1536 \
# --batch_size 1


# for chunked-kgat-big-4-layer with 1536 att hidden size
# CUDA_VISIBLE_DEVICES=12 python test.py --outdir ./output/ \
# --test_path ../data/standard_srl/srl_bert_eval.json \
# --bert_pretrain ../bert_base \
# --checkpoint ../checkpoint/chunked-kgat-big-4-layers/model.best.pt \
# --name chunked-dev-train-att-big-4-layers.json \
# --attn_hidden_size 1536 \
# --batch_size 1 \
# --n_attn_layer 4

# chunked-kgat att with label smoothing
CUDA_VISIBLE_DEVICES=12 python test.py --outdir ./output/ \
--test_path ../data/standard_srl/srl_bert_eval.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/chunked-kgat-label-smoothing/model.best.pt \
--name chunked-dev-label-smoothing.json \
--attn_hidden_size 1536 \
--batch_size 1 \
--n_attn_layer 4