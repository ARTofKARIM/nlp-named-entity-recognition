"""NER model training."""
import torch
from torch.utils.data import DataLoader, TensorDataset
from seqeval.metrics import classification_report, f1_score

class NERTrainer:
    def __init__(self, model, id2label, device="cpu"):
        self.model = model.to(device)
        self.id2label = id2label
        self.device = device

    def train(self, X_train, y_train, epochs=20, batch_size=32, lr=0.001):
        X_t = torch.LongTensor(X_train)
        y_t = torch.LongTensor(y_train)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    def evaluate(self, X_test, y_test, lengths=None):
        self.model.eval()
        X_t = torch.LongTensor(X_test).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            preds = logits.argmax(dim=-1).cpu().numpy()
        true_labels, pred_labels = [], []
        for i in range(len(y_test)):
            t, p = [], []
            for j in range(len(y_test[i])):
                if y_test[i][j] != 0:
                    t.append(self.id2label.get(y_test[i][j], "O"))
                    p.append(self.id2label.get(preds[i][j], "O"))
            true_labels.append(t)
            pred_labels.append(p)
        print(classification_report(true_labels, pred_labels))
        return f1_score(true_labels, pred_labels)
