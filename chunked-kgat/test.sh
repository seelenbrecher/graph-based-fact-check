CUDA_VISIBLE_DEVICES=9 python test.py --outdir ./output/ \
--test_path ../data/standard_srl/srl_bert_eval.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/chunked-kgat/model.best.pt \
--name chunked-dev.json \
--batch_size 1 \
--max_len 60


CUDA_VISIBLE_DEVICES=9 python test.py --outdir ./output/ \
--test_path ../data/standard_srl/srl_bert_test.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/chunked-kgat/model.best.pt \
--name chunked-test.json \
--batch_size 1 \
--max_len 60

