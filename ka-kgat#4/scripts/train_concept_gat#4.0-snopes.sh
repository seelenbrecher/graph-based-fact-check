# python train.py --outdir ../checkpoint/kgat \
# --train_path ../data/bert_train.json \
# --valid_path ../data/bert_dev.json \
# --bert_pretrain ../bert_base \
# --postpretrain ../pretrain/save_model/model.best.pt

NAME=ka-kgat-concept-gat\#4.0.snopes

rm ../checkpoint/$NAME/train_log.txt

# # use concept + gat
# CUDA_VISIBLE_DEVICES=9 python train.py --outdir ../checkpoint/$NAME \
# --train_path ../data/fever_concepts_with_sent_labels/train.json \
# --valid_path ../data/fever_concepts_with_sent_labels/eval.json \
# --bert_pretrain ../bert_base \
# --postpretrain ../checkpoint/pretrain/model.best.pt \
# --ner_srl_train_path ../data/fever_srl_and_ner#1/train.json \
# --ner_srl_eval_path ../data/fever_srl_and_ner#1/eval.json \
# --ner_srl_test_path ../data/fever_srl_and_ner#1/test.json \
# --use_concept \
# --span_use_gat \
# --num_train_epochs 10 \
# --span_gat_dropout 0.0

# CUDA_VISIBLE_DEVICES=9 python test.py --outdir ./output/ \
# --test_path ../data/fever_with_concepts/bert_eval_concept.json \
# --bert_pretrain ../bert_base \
# --checkpoint ../checkpoint/$NAME/model.best.pt \
# --ner_srl_train_path ../data/fever_srl_and_ner#1/train.json \
# --ner_srl_eval_path ../data/fever_srl_and_ner#1/eval.json \
# --ner_srl_test_path ../data/fever_srl_and_ner#1/test.json \
# --use_concept \
# --span_use_gat \
# --span_gat_dropout 0.0 \
# --name $NAME-dev.json

CUDA_VISIBLE_DEVICES=8 python train.py --outdir ../checkpoint/$NAME \
--train_path ../data/snopes_concepts/train.json \
--valid_path ../data/snopes_concepts/eval.json \
--bert_pretrain ../bert_base \
--postpretrain ../checkpoint/pretrain/model.best.pt \
--ner_srl_train_path ../data/snopes_srl_and_ner/train.json \
--ner_srl_eval_path ../data/snopes_srl_and_ner/eval.json \
--ner_srl_test_path ../data/snopes_srl_and_ner/test.json \
--use_concept \
--span_use_gat \
--num_train_epochs 10 \
--no_evidence_selection \
--eval_step 50 \
--gradient_accumulation_steps 2 \
--train_batch_size 4 \
--span_gat_dropout 0.0

# CUDA_VISIBLE_DEVICES=4 python test.py --outdir ./output/ \
# --test_path ../data/snopes_concepts/eval.json \
# --bert_pretrain ../bert_base \
# --checkpoint ../checkpoint/$NAME/model.best.pt \
# --ner_srl_train_path ../data/snopes_srl_and_ner/train.json \
# --ner_srl_eval_path ../data/snopes_srl_and_ner/eval.json \
# --ner_srl_test_path ../data/snopes_srl_and_ner/test.json \
# --use_concept \
# --span_use_gat \
# --span_gat_dropout 0.0 \
# --name $NAME-dev.json

# python fever_score_test.py --predicted_labels ./output/$NAME-dev.json  --predicted_evidence ../data/bert_eval.json --actual ../data/dev_eval.json

# CUDA_VISIBLE_DEVICES=15 python test.py --outdir ./output/ \
#  --test_path ../data/fever_with_concepts/bert_test_concept.json \
#  --bert_pretrain ../bert_base \
#  --checkpoint ../checkpoint/$NAME/model.best.pt \
#  --use_concept \
#  --span_use_gat \
#  --span_gat_dropout 0.0 \
#  --name $NAME-test.json
