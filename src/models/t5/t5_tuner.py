import torch
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import numpy as np
from src.models.t5.pytorch_dataset import QuestionGenerationDataset
import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
)


t5_tokenizer = T5Tokenizer.from_pretrained('t5-small', model_max_length=512)
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')

train_path = '../../../data/train_squad.parquet'
validation_path = '../../../data/validation_squad.parquet'
train_dataset = QuestionGenerationDataset(t5_tokenizer, train_path)
validation_dataset = QuestionGenerationDataset(t5_tokenizer, validation_path)

train_sample = train_dataset[50]
decoded_train_input = t5_tokenizer.decode(train_sample['source_ids'])
decoded_train_output = t5_tokenizer.decode(train_sample['target_ids'])

class T5Tuner(pl.LightningModule):

    def __init__(self, t5model, t5tokenizer, learning_rate=3e-4, batchsize=4, weight_decay=0.0):
        super().__init__()
        self.model = t5model
        self.tokenizer = t5tokenizer
        self.learning_rate = learning_rate
        self.batch_size = batchsize
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            eps=1e-8,
            weight_decay=self.weight_decay
        )
        return optimizer

    def forward(self, input_ids, attention_mask=None,
                decoder_attention_mask=None,
                lm_labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )

        loss = outputs[0]
        print(f"Training Loss: {loss.item()}")
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )

        loss = outputs[0]
        self.log("val_loss", loss)

        predictions = self.model.generate(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            max_length=self.tokenizer.model_max_length
        )
        decoded_preds = [self.tokenizer.decode(p, skip_special_tokens=True) for p in predictions]
        decoded_targets = [self.tokenizer.decode(t, skip_special_tokens=True) for t in batch["target_ids"]]

        # Compute BLEU
        bleu_scores = [sentence_bleu([target.split()], pred.split()) for pred, target in
                       zip(decoded_preds, decoded_targets)]
        avg_bleu = np.mean(bleu_scores)
        self.log("val_bleu", avg_bleu)

        # Compute ROUGE
        rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = [rouge.score(target, pred) for pred, target in zip(decoded_preds, decoded_targets)]
        avg_rouge1 = np.mean([s['rouge1'].fmeasure for s in rouge_scores])
        avg_rouge2 = np.mean([s['rouge2'].fmeasure for s in rouge_scores])
        avg_rougeL = np.mean([s['rougeL'].fmeasure for s in rouge_scores])

        self.log("val_rouge1", avg_rouge1)
        self.log("val_rouge2", avg_rouge2)
        self.log("val_rougeL", avg_rougeL)

        # Compute Perplexity
        perplexity = torch.exp(loss)
        self.log("val_perplexity", perplexity)

        return {"val_loss": loss, "val_bleu": avg_bleu, "val_rouge1": avg_rouge1}

    def test_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )

        loss = outputs[0]

        # Generate predictions
        predictions = self.model.generate(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            max_length=self.tokenizer.model_max_length
        )

        # Decode predictions and targets
        decoded_preds = [self.tokenizer.decode(p, skip_special_tokens=True) for p in predictions]
        decoded_targets = [self.tokenizer.decode(t, skip_special_tokens=True) for t in batch["target_ids"]]

        # Calculate BLEU scores
        bleu_scores = [sentence_bleu([target.split()], pred.split())
                       for pred, target in zip(decoded_preds, decoded_targets)]
        avg_bleu = np.mean(bleu_scores)

        # Calculate ROUGE scores
        rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = [rouge.score(target, pred) for pred, target in zip(decoded_preds, decoded_targets)]
        avg_rouge1 = np.mean([s['rouge1'].fmeasure for s in rouge_scores])
        avg_rouge2 = np.mean([s['rouge2'].fmeasure for s in rouge_scores])
        avg_rougeL = np.mean([s['rougeL'].fmeasure for s in rouge_scores])

        self.log('test_loss', loss)
        self.log('test_bleu', avg_bleu)
        self.log('test_rouge1', avg_rouge1)
        self.log('test_rouge2', avg_rouge2)
        self.log('test_rougeL', avg_rougeL)

        return {
            "test_loss": loss.item(),
            "test_bleu": avg_bleu,
            "test_rouge1": avg_rouge1,
            "test_rouge2": avg_rouge2,
            "test_rougeL": avg_rougeL
        }

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.batch_size,
                          num_workers=2)

    def val_dataloader(self):
        return DataLoader(validation_dataset,
                          batch_size=self.batch_size,
                          num_workers=2)

    def test_dataloader(self):
        test_dataset = QuestionGenerationDataset(self.tokenizer, '../../../data/test_squad.parquet')
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=2)
