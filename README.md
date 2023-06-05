# run-NLP-model
> The script to run NLP model on SQuAD, sst2 and QQP dataset 

Try to run the command:

### SQuAD

```shell
python3 squad.py <model> <dataset> <result> <amount>
```

where:

- `model` is the model you want to use.  `t5` or `DeBERTa` can be chosen
- `dataset` is the json file of SQuAD dataset
- `result` is the output file
- `amount` is the amount of QA you want to run 

### sst2

```shell
python3 sst.py <model> <dataset> <result> <amount>
```

where:

- `model` is the model you want to use.  `t5` or `DeBERTa` or `distilbert` can be chosen
- `dataset` is the txt file of sst2 dataset
- `result` is the output file
- `amount` is the amount of sentences you want to run

### QQP

```shell
python3 qqp.py <model> <dataset> <result> <amount>
```

where:

- `model` is the model you want to use.  `t5` or `DeBERTa` can be chosen
- `dataset` is the tsv file of QQP dataset
- `result` is the output file
- `amount` is the amount of sentence pairs you want to run 
