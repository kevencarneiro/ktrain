import os
import pickle

import re

import pandas as pd
from sklearn.model_selection import train_test_split

from ktrain.text import AnswerExtractor

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# try:
#     nlp = spacy.load('pt_core_news_sm')
# except Exception:
#     spacy.cli.download("pt_core_news_sm")
#     nlp = spacy.load('pt_core_news_sm')

INDEXDIR = 'rdai-qa'
MODEL_NAME = 'pierreguillou/bert-base-cased-squad-v1.1-portuguese'
BERT_EMB_MODEL = 'neuralmind/bert-base-portuguese-cased'

max_context_length = 350
include_no_answer = False
include_whitespace_only = False
normalize_punctuation = False

# qa = SimpleQA(INDEXDIR, model_name=MODEL_NAME, bert_emb_model=BERT_EMB_MODEL)
#
# qa.ask('quem descobriu o brasil?')

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] =
# creds, _ = default()
#
# gc = gspread.oauth() #authorize(creds)
# gc.open_by_key('16qbWXYeh0Dza382dDdINYDlQ4g5CMdtL2h69NFIH8oM')
# worksheet = gc.open_by_key('16qbWXYeh0Dza382dDdINYDlQ4g5CMdtL2h69NFIH8oM').get_worksheet(0)
# rows = worksheet.get_all_values()
# df = pd.DataFrame.from_records(rows[1:], columns=rows[0])

df = pd.read_csv('data.csv')
df.fillna('', inplace=True)

if not include_whitespace_only:
    idx_whitespace = df.index[df['Resposta'].apply(lambda x: (not x) or x.isspace())]
    df.drop(index=idx_whitespace, inplace=True)
if not include_no_answer:
    idx_no_answer = df.index[df['Resposta'].apply(lambda x: x.lower() == 'não há resposta')]
    df.drop(index=idx_no_answer, inplace=True)
# if max_context_length:
#     idx_large_context = df.index[df['Contexto'].apply(lambda x: len(nlp.tokenizer(x)) > max_context_length)]
#     df.drop(index=idx_large_context, inplace=True)


def format_string(string: str):
    regex_markers = r"((\s+|^)\(?\d{0,2}[[iIvVxX]*\w?[|*.\-•)]+\s+)"

    string = string.strip()
    string = string.replace('\r\n', '\n')
    string = string.replace('\n', ' ')
    if normalize_punctuation:
        string = re.sub(regex_markers, r" * ", string)
    string = re.sub(r"\n+", r"\n", string)
    string = re.sub(r"[ \t]+", r" ", string)
    return string


df['Título'] = df['Título'].apply(format_string)
df['Seção'] = df['Seção'].apply(format_string)
df['Contexto'] = df['Contexto'].apply(format_string)
df['Pergunta'] = df['Pergunta'].apply(format_string).astype('category')
df['Resposta'] = df['Resposta'].apply(format_string)

df = df[df['Pergunta'].astype(bool)]
df = df[df['Contexto'].astype(bool)]

questions = []

for keys, values in df.groupby(['Contexto', 'Pergunta']):
    questions.append({
        'question': values['Pergunta'].iloc[0],
        'context': values['Contexto'].iloc[0],
        'answers': values['Resposta'].values
    })

train_questions, test_questions = train_test_split(questions, stratify=[q['question'] for q in questions], random_state=42)

ae = AnswerExtractor(BERT_EMB_MODEL)
ae.finetune(data=train_questions, epochs=8, batch_size=8, learning_rate=3e-5)

evaluation_results = ae.qa.evaluate_squad(test_questions)
print(evaluation_results)

ae.qa.predict_squad(documents=questions[0]['context'], question=questions[0]['question'])