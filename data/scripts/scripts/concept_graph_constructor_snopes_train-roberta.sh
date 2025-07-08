# python construct_concept_graph.py --input ../snopes_with_concepts.bk2/train.json \
#  --output ../snopes_with_concepts_and_graph_roberta.bk3/train \
#  --bert_pretrain ../../checkpoint/roberta_large_mlm \
#  --roberta
 
#  python construct_concept_graph.py --input ../snopes_with_concepts.bk2/dev.json \
#  --output ../snopes_with_concepts_and_graph_roberta.bk3/dev \
#  --bert_pretrain ../../checkpoint/roberta_large_mlm \
#  --roberta
 
#  python construct_concept_graph.py --input ../snopes_with_concepts.bk2/test.json \
#  --output ../snopes_with_concepts_and_graph_roberta.bk3/test \
#  --bert_pretrain ../../checkpoint/roberta_large_mlm \
#  --roberta


python construct_concept_graph.py --input ../snopes_with_concepts.bk/train.json \
 --output ../snopes_with_concepts_and_graph_roberta.bk_full/train \
 --bert_pretrain ../../checkpoint/roberta_large_mlm \
 --roberta
 
 python construct_concept_graph.py --input ../snopes_with_concepts.bk/dev.json \
 --output ../snopes_with_concepts_and_graph_roberta.bk_full/dev \
 --bert_pretrain ../../checkpoint/roberta_large_mlm \
 --roberta
 
 python construct_concept_graph.py --input ../snopes_with_concepts.bk/test.json \
 --output ../snopes_with_concepts_and_graph_roberta.bk_full/test \
 --bert_pretrain ../../checkpoint/roberta_large_mlm \
 --roberta
