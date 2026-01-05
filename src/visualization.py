"""NER visualization."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class NERVisualizer:
    def __init__(self, output_dir="results/"):
        self.output_dir = output_dir

    def highlight_entities(self, tokens, labels):
        colors = {"PER": "\033[91m", "ORG": "\033[94m", "LOC": "\033[92m", "MISC": "\033[93m"}
        reset = "\033[0m"
        result = []
        for token, label in zip(tokens, labels):
            if label.startswith("B-") or label.startswith("I-"):
                etype = label[2:]
                color = colors.get(etype, "")
                result.append(f"{color}[{token}/{label}]{reset}")
            else:
                result.append(token)
        return " ".join(result)

    def plot_entity_distribution(self, label_sequences, save=True):
        from collections import Counter
        entity_counts = Counter()
        for seq in label_sequences:
            for label in seq:
                if label != "O":
                    entity_counts[label] += 1
        if not entity_counts:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        labels, counts = zip(*entity_counts.most_common(15))
        ax.bar(labels, counts, color="steelblue")
        ax.set_title("Entity Type Distribution")
        plt.xticks(rotation=45, ha="right")
        if save:
            fig.savefig(f"{self.output_dir}entity_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
