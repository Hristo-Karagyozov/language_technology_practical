import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
import optuna
import json
from src.models.t5.pytorch_dataset import QuestionGenerationDataset
from src.models.t5.t5_tuner import T5Tuner
import pytorch_lightning as pl
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
)

device = 'cuda' if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('medium')


def create_pandas(data):
    result_df = pd.DataFrame(columns=['context', 'question'])
    for index, val in enumerate(tqdm(data)):
        passage = val['context']
        question = val['question']
        result_df.loc[index] = [passage] + [question]

    return result_df


squad_dataset = load_dataset('squad', split='train[:50%]')  # Use 50% of the dataset

# 50% of SQuAD into 80% train, 10% validation, 10% test
train_val_test_split = squad_dataset.train_test_split(test_size=0.2)  # 80% train, 20% remaining
val_test_split = train_val_test_split['test'].train_test_split(test_size=0.5)  # Split rem. 20% into 10% val, 10% test

train_dataset = train_val_test_split['train']  # 80%
validation_dataset = val_test_split['train']  # 10%
test_dataset = val_test_split['test']  # 10%

pandas_train = create_pandas(train_dataset)
pandas_validation = create_pandas(validation_dataset)
pandas_test = create_pandas(test_dataset)

# Saving data for future use
pandas_train.to_parquet('../../../data/train_squad.parquet')
pandas_validation.to_parquet('../../../data/validation_squad.parquet')
pandas_test.to_parquet('../../../data/test_squad.parquet')


t5_tokenizer = T5Tokenizer.from_pretrained('t5-small', model_max_length=512)
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')

train_path = '../../../data/train_squad.parquet'
validation_path = '../../../data/validation_squad.parquet'
train_dataset = QuestionGenerationDataset(t5_tokenizer, train_path)
validation_dataset = QuestionGenerationDataset(t5_tokenizer, validation_path)

train_sample = train_dataset[50]  #
decoded_train_input = t5_tokenizer.decode(train_sample['source_ids'])
decoded_train_output = t5_tokenizer.decode(train_sample['target_ids'])

def objective(trial):
    # hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)

    # Create model with trial hyperparameters
    model = T5Tuner(
        t5_model,
        t5_tokenizer,
        learning_rate=learning_rate,
        batchsize=batch_size,
        weight_decay=weight_decay
    )

    #  trainer with pruning callback
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator=device,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                mode="min"
            )
        ],
        enable_checkpointing=False,
        logger=True  # Enable logging
    )

    # Train model
    trainer.fit(model)

    # Return the validation loss
    val_loss = trainer.callback_metrics.get("val_loss")
    return val_loss.item()


def main():
    study = optuna.create_study(direction="minimize")

    study.optimize(objective, n_trials=20)

    # Only proceed if we got any successful trials
    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Train final model with best parameters
    final_model = T5Tuner(
        t5_model,
        t5_tokenizer,
        learning_rate=study.best_trial.params["learning_rate"],
        batchsize=study.best_trial.params["batch_size"],
        weight_decay=study.best_trial.params["weight_decay"]
    )

    final_trainer = pl.Trainer(
        max_epochs=5,
        accelerator=device,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                mode="min"
            )
        ],
        enable_checkpointing=True
    )

    final_trainer.fit(final_model)
    test_results = final_trainer.test(final_model)

    with open('final_test_results.json', 'w') as f:
        json.dump(test_results[0], f, indent=4)

    print("\nFinal Test Results:")
    for metric, value in test_results[0].items():
        print(f"{metric}: {value}")

    # Save the model
    final_model.model.save_pretrained('t5_final_model')
    t5_tokenizer.save_pretrained('t5_final_tokenizer')

    # Save study results
    study_df = study.trials_dataframe()
    study_df.to_csv('optuna_study_results.csv')

    # some example predictions
    test_batch = next(iter(final_model.test_dataloader()))
    predictions = final_model.model.generate(
        input_ids=test_batch["source_ids"][:5].to(device),
        attention_mask=test_batch["source_mask"][:5].to(device),
        max_length=final_model.tokenizer.model_max_length
    )

    examples = []
    for i in range(5):
        example = {
            "context": final_model.tokenizer.decode(test_batch["source_ids"][i], skip_special_tokens=True),
            "target": final_model.tokenizer.decode(test_batch["target_ids"][i], skip_special_tokens=True),
            "prediction": final_model.tokenizer.decode(predictions[i], skip_special_tokens=True)
        }
        examples.append(example)

    with open('example_predictions.json', 'w') as f:
        json.dump(examples, f, indent=4)


if __name__ == "__main__":
    main()
