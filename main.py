import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys
import transformers
from transformers import pipeline

OPTS = None

def parse_args():
    parser = argparse.ArgumentParser(
        'Script for t5(https://huggingface.co/mrm8488/t5-base-finetuned-squadv2) and DeBERTa(https://huggingface.co/deepset/deberta-v3-base-squad2).\n \
        dataset: SQuAD version 2.0.\n \
        nlp task: MRC.')
    parser.add_argument('model', metavar='model', 
                        help='Model you want to use (t5 or DeBERTa).', type=str)
    parser.add_argument('data_file', metavar='data.json',
                        help='Input data JSON file.')
    parser.add_argument('out_file', metavar='result.txt',
                        help='Write result to file.')
    parser.add_argument('amount', metavar='amount',
                        help='Amount of QA you want to test.', type=int)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def t5_get_answer(question, context, nlp):
    input_text = f"question: {question}  context: {context}"
    res = nlp(input_text)
    if 'generated_text' in res[0]:
        return res[0]['generated_text']
    else:
        return ''

def deberta_get_answer(question, context, nlp):
    QA_input = {
        'question': question,
        'context': context
    }
    res = nlp(QA_input)
    if 'answer' in res:
        return res['answer']
    else:
        return ''

def get_answer(question, context, nlp):
    if OPTS.model == 't5':
        return t5_get_answer(question, context, nlp)
    elif OPTS.model == 'DeBERTa':
        return deberta_get_answer(question, context, nlp)
    else:
        return ''

def main(nlp):
    with open(OPTS.data_file) as f:
        dataset_json = json.load(f)
        dataset = dataset_json['data']
    i = 0
    with open(OPTS.out_file, 'w') as f:
        for paragraphs in dataset:
            for paragraph in paragraphs['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    answer = get_answer(question, context, nlp)
                    i += 1
                    print(f'complete: {i} answer: {answer}')
                    f.write(answer+'\n')
                    if (i >= OPTS.amount):
                        return 

if __name__ == '__main__':
    OPTS = parse_args()
    if OPTS.model == 't5':
        model_name = "mrm8488/t5-base-finetuned-squadv2"
        nlp = pipeline('text2text-generation', model=model_name, tokenizer=model_name)
    elif OPTS.model == 'DeBERTa':
        model_name = "deepset/deberta-v3-base-squad2"
        nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    else:
        NameError
    main(nlp)