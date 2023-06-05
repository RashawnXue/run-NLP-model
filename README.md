# run-NLP-model
> The script to run NLP model on SQuAD and sst2 dataset

Try to run the command:

### SQuAD

```shell
python3 main.py <model> <dataset> <result> <amount>
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
