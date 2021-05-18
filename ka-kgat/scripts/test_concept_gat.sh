# python test.py --outdir ./output \
# --test_path ../data/bert_eval.json \
# --bert_pretrain ../bert_base \
# --checkpoint ../checkpoint/kgat/model.best.pt \
# --name dev.json

# python test.py --outdir ./output \
# --test_path ../data/bert_test.json \
# --bert_pretrain ../bert_base \
# --checkpoint ../checkpoint/kgat/model.best.pt \
# --name test.json



CUDA_VISIBLE_DEVICES=8 python test.py --outdir ./output/ \
--test_path ../data/fever_with_concepts/bert_eval_concept.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/ka-kgat-concept-gat/model.best.pt \
--use_concept \
--span_use_gat \
--name ka-kgat-concept-gat-dev.json

# CUDA_VISIBLE_DEVICES=8 python test.py --outdir ./output/ \
# --test_path ../data/fever_with_concepts/bert_test_concept.json \
# --bert_pretrain ../bert_base \
# --checkpoint ../checkpoint/ka-kgat-concept-only/model.best.pt \
# --name ka-kgat-concept-only-test.json
