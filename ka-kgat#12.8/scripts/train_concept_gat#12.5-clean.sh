# python train.py --outdir ../checkpoint/kgat \
# --train_path ../data/bert_train.json \
# --valid_path ../data/bert_dev.json \
# --bert_pretrain ../bert_base \
# --postpretrain ../pretrain/save_model/model.best.pt
# with full data
NAME=ka-kgat-concept-gat\#12.5.clean

rm ../checkpoint/$NAME/train_log.txt

# use concept + gat
CUDA_VISIBLE_DEVICES=7 python train.py --outdir ../checkpoint/$NAME \
--train_path ../data/snopes_with_concepts_and_graph_clean.5.7/train.json \
--valid_path ../data/snopes_with_concepts_and_graph_clean.5.7/test.json \
--bert_pretrain ../bert_base \
--postpretrain ../checkpoint/pretrain/model.best.pt \
--use_concept \
--span_use_gat \
--span_gat_add_skip_conn \
--eval_step 50 \
--num_train_epochs 50 \
--span_gat_dropout 0.0 \
--with_f1

CUDA_VISIBLE_DEVICES=7 python test.py --outdir ./output/ \
 --test_path ../data/snopes_with_concepts_and_graph_clean.5.7/test.json \
 --bert_pretrain ../bert_base \
 --checkpoint ../checkpoint/$NAME/model.best.pt \
 --use_concept \
 --span_use_gat \
 --span_gat_dropout 0.0 \
 --span_gat_add_skip_conn \
 --name $NAME-dev.json

python ukp_score_test.py --predicted_labels ./output/$NAME-dev.json  --predicted_evidence ../data/bert_eval.json --actual ../data/snopes_with_concepts_and_graph_clean.5.7/test.json
