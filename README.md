# Incorporating External Knowledge for Evidence-based Fact Verification

Source codes for: [Incorporating External Knowledge for Evidence-based Fact
Verification](https://dl.acm.org/doi/pdf/10.1145/3487553.3524622)

## Requirements
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Data and checkpoint
* `bert_base` pre-trained checkpoint available [here](https://thunlp.oss-cn-qingdao.aliyuncs.com/KernelGAT/FEVER/KernelGAT.zip) and put it under `bert_base` directory
* `roberta_large` pre-trained checkpoint available [here](https://thunlp.oss-cn-qingdao.aliyuncs.com/KernelGAT/FEVER/KernelGAT_roberta_large.zip), and put it under `roberta_large_mlm` directory
* FEVER dataset can be obtained [here](https://fever.ai/dataset/fever.html)
* UKP Snopes Corpus dataset can be obtained [here](https://tudatalib.ulb.tu-darmstadt.de/items/0d997a9b-72e9-4e72-b5c7-899265c5d897)


## Fine-tune model
* To fine-tune `bert_base` model, run `bash scripts/train_bert.sh`
* To fine-tune `roberta_large` model, run `bash scripts/train_roberta.sh`

## Citation
```
@inproceedings{barik2022incorporating,
  title={Incorporating external knowledge for evidence-based fact verification},
  author={Barik, Anab Maulana and Hsu, Wynne and Lee, Mong Li},
  booktitle={Companion Proceedings of the Web Conference 2022},
  pages={429--437},
  year={2022}
}
```