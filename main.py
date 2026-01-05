"""Main NER pipeline."""
import yaml
from src.data_loader import CoNLLLoader
from src.bilstm_model import BiLSTMNER
from src.trainer import NERTrainer

def main():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    loader = CoNLLLoader(config["data"]["labels"])
    train_sents, train_labels = loader.load_conll(config["data"]["train_path"])
    vocab = loader.build_vocab(train_sents)
    X_train = loader.encode_sentences(train_sents, vocab)
    y_train = loader.encode_labels(train_labels)
    model = BiLSTMNER(len(vocab), config["model"]["embedding_dim"], config["model"]["hidden_dim"],
                       len(config["data"]["labels"]), config["model"]["dropout"])
    trainer = NERTrainer(model, loader.id2label)
    trainer.train(X_train, y_train, config["model"]["epochs"], config["model"]["batch_size"])
    print("Training complete.")

if __name__ == "__main__":
    main()
