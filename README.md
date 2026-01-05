# Named Entity Recognition

A BiLSTM-based NER system for extracting named entities (Person, Organization, Location) from text with CoNLL format support.

## Architecture
```
nlp-named-entity-recognition/
├── src/
│   ├── data_loader.py    # CoNLL format loading and encoding
│   ├── bilstm_model.py   # BiLSTM-CRF model
│   ├── trainer.py         # Training loop with seqeval metrics
│   └── visualization.py   # Entity highlighting and distribution plots
├── config/config.yaml
├── tests/test_loader.py
└── main.py
```

## Entity Types
| Entity | Tag | Example |
|--------|-----|---------|
| Person | PER | John Smith |
| Organization | ORG | Google |
| Location | LOC | Paris |
| Miscellaneous | MISC | Nobel Prize |

## Installation
```bash
git clone https://github.com/mouachiqab/nlp-named-entity-recognition.git
cd nlp-named-entity-recognition && pip install -r requirements.txt
python main.py
```

## Technologies
- Python 3.9+, PyTorch, spaCy, seqeval, transformers







