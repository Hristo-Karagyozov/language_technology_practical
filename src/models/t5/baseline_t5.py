from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import numpy as np
import json
import torch
from src.models.t5.pytorch_dataset import QuestionGenerationDataset

device = 'cuda' if torch.cuda.is_available() else "cpu"

# Load the pre-trained model and tokenizer
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small', model_max_length=512)
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)

# Load the test dataset
test_dataset = QuestionGenerationDataset(t5_tokenizer, '../../../data/test_squad.parquet')
test_dataloader = DataLoader(test_dataset, batch_size=4, num_workers=2)

# Set the model to evaluation mode
t5_model.eval()

# Metrics calculation
all_predictions = []
all_targets = []

rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
bleu_scores = []
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch["source_ids"].to(device)
        attention_mask = batch["source_mask"].to(device)

        # Generate predictions
        predictions = t5_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=t5_tokenizer.model_max_length
        )

        # Decode predictions and targets
        decoded_preds = [t5_tokenizer.decode(p, skip_special_tokens=True) for p in predictions]
        decoded_targets = [t5_tokenizer.decode(t, skip_special_tokens=True) for t in batch["target_ids"]]

        all_predictions.extend(decoded_preds)
        all_targets.extend(decoded_targets)

        # Compute BLEU and ROUGE for each sample
        for pred, target in zip(decoded_preds, decoded_targets):
            bleu_scores.append(sentence_bleu([target.split()], pred.split()))

            rouge_scores = rouge.score(target, pred)
            rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
            rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
            rougeL_scores.append(rouge_scores['rougeL'].fmeasure)

# Compute average scores
avg_bleu = np.mean(bleu_scores)
avg_rouge1 = np.mean(rouge1_scores)
avg_rouge2 = np.mean(rouge2_scores)
avg_rougeL = np.mean(rougeL_scores)

# Save results
results = {
    "bleu": avg_bleu,
    "rouge1": avg_rouge1,
    "rouge2": avg_rouge2,
    "rougeL": avg_rougeL
}

with open('baseline_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nBaseline Results:")
for metric, value in results.items():
    print(f"{metric}: {value}")

# Save example predictions
examples = []
for i in range(5):
    example = {
        "target": all_targets[i],
        "prediction": all_predictions[i]
    }
    examples.append(example)

with open('baseline_example_predictions.json', 'w') as f:
    json.dump(examples, f, indent=4)
