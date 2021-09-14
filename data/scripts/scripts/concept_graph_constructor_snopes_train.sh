# python construct_concept_graph.py --input ../snopes_with_concepts.bk2/train.json \
#  --output ../snopes_with_concepts_and_graph.bk3/train.json
 
#  python construct_concept_graph.py --input ../snopes_with_concepts.bk2/dev.json \
#  --output ../snopes_with_concepts_and_graph.bk3/dev.json
 
#  python construct_concept_graph.py --input ../snopes_with_concepts.bk2/test.json \
#  --output ../snopes_with_concepts_and_graph.bk3/test.json


python construct_concept_graph.py --input ../snopes_with_concepts.bk/train.json \
 --output ../snopes_with_concepts_and_graph.bk_full/train.json
 
 python construct_concept_graph.py --input ../snopes_with_concepts.bk/dev.json \
 --output ../snopes_with_concepts_and_graph.bk_full/dev.json
 
 python construct_concept_graph.py --input ../snopes_with_concepts.bk/test.json \
 --output ../snopes_with_concepts_and_graph.bk_full/test.json
