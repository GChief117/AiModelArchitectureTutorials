import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from collections import Counter

# ================================================================
# 🚀 PERPLEXITY SCORE (Measures Predictability)
# ================================================================

def calculate_perplexity(model, test_loader, device="cuda"):
    """
    Computes perplexity (PPL) on test data.
    - Lower perplexity means the model predicts tokens more accurately.

    Parameters:
    - model: The trained Transformer model.
    - test_loader: DataLoader with test data.
    - device: "cuda" or "cpu".

    Returns:
    - Perplexity score (lower = better).
    """
    model.to(device)
    model.eval()
    total_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token (0)

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)  # Model predicts token probabilities
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), batch.view(-1))  # Compute loss
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    perplexity = torch.exp(torch.tensor(avg_loss))  # e^(cross-entropy loss)

    print(f"Perplexity Score: {perplexity:.2f} (Lower is better)")
    return perplexity

# ================================================================
# 🚀 BLEU SCORE (Measures Similarity to Reference Texts)
# ================================================================

def calculate_bleu(reference_texts, generated_texts):
    """
    Computes BLEU score between reference texts and model-generated texts.
    - BLEU measures how close generated text is to human-written text.

    Parameters:
    - reference_texts: List of ground truth text samples.
    - generated_texts: List of model-generated samples.

    Returns:
    - Average BLEU score (higher = better).
    """
    scores = []
    for ref, gen in zip(reference_texts, generated_texts):
        reference = [ref.split()]  # Reference text split into words
        candidate = gen.split()  # Model-generated text split into words
        score = sentence_bleu(reference, candidate)  # Compute BLEU score
        scores.append(score)

    avg_bleu = sum(scores) / len(scores)  # Compute average BLEU across samples
    print(f"BLEU Score: {avg_bleu:.4f} (Higher is better)")
    return avg_bleu

# ================================================================
# 🚀 ROUGE SCORE (Measures Recall & Overlap with Reference Texts)
# ================================================================

def calculate_rouge(reference_texts, generated_texts):
    """
    Computes ROUGE scores for model evaluation.
    - ROUGE measures how much of the reference text appears in the generated text.

    Parameters:
    - reference_texts: List of human-written reference texts.
    - generated_texts: List of model-generated outputs.

    Returns:
    - ROUGE scores (Higher is better).
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, gen) for ref, gen in zip(reference_texts, generated_texts)]

    # Compute average ROUGE scores across all test samples
    avg_rouge = {
        'rouge1': sum(s['rouge1'].fmeasure for s in scores) / len(scores),
        'rouge2': sum(s['rouge2'].fmeasure for s in scores) / len(scores),
        'rougeL': sum(s['rougeL'].fmeasure for s in scores) / len(scores)
    }

    print(f"ROUGE Scores: {avg_rouge}")
    return avg_rouge

# ================================================================
# 🚀 TEXT DIVERSITY SCORE (Measures Uniqueness of Generated Text)
# ================================================================

def measure_text_diversity(generated_texts, n=2):
    """
    Measures diversity in model-generated text by analyzing distinct n-grams.
    - Ensures that generated text is not repetitive or generic.

    Parameters:
    - generated_texts: List of generated text outputs.
    - n: N-gram size (default = 2 for bigrams).

    Returns:
    - Distinct n-gram ratio (higher = more diverse).
    """
    all_ngrams = []
    total_ngrams = 0

    for text in generated_texts:
        words = text.split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        all_ngrams.extend(ngrams)
        total_ngrams += len(ngrams)

    unique_ngrams = len(set(all_ngrams))
    diversity_score = unique_ngrams / total_ngrams if total_ngrams > 0 else 0

    print(f"Diversity Score (Distinct-{n} Ratio): {diversity_score:.4f} (Higher is better)")
    return diversity_score

# ================================================================
# 🚀 RUN THE FULL BACKTESTING PIPELINE
# ================================================================

if __name__ == "__main__":
    """
    Runs all backtesting benchmarks for a trained language model.
    - Loads reference and generated text.
    - Computes Perplexity, BLEU, ROUGE, and Diversity Scores.
    """

    # Example reference and generated texts (For real evaluation, use a larger dataset)
    reference_texts = ["The cat sat on the mat.", "Artificial intelligence is advancing rapidly."]
    generated_texts = ["The cat lay on the rug.", "AI is progressing very fast."]

    # Load trained model and test data
    model = TransformerModel(vocab_size=50000)  # Load your trained model
    model.load_state_dict(torch.load("best_transformer_model.pth"))  # Load best model
    model.eval()

    # Define test DataLoader
    test_loader = DataLoader(test_dataset, batch_size=16)

    print("\n===== Backtesting Results =====\n")

    # Compute Perplexity (Lower is Better)
    perplexity = calculate_perplexity(model, test_loader, device="cuda")

    # Compute BLEU Score (Higher is Better)
    bleu_score = calculate_bleu(reference_texts, generated_texts)

    # Compute ROUGE Score (Higher is Better)
    rouge_scores = calculate_rouge(reference_texts, generated_texts)

    # Compute Diversity Score (Higher is Better)
    diversity_score = measure_text_diversity(generated_texts)
