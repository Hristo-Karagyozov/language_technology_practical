import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
import copy


class QuestionGenerationDataset(Dataset):
    def __init__(self, tokenizer, filepath, max_len_inp=512, max_len_out=96):
        self.path = filepath

        self.passage_column = "context"
        self.question = "question"

        self.data = pd.read_parquet(self.path)

        self.max_len_input = max_len_inp
        self.max_len_output = max_len_out
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        # Squeezing here removes the batch dimension added with the tokenization in _build()
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        # Attention masks extracted from the arrays constructed in _build()
        src_mask = self.inputs[index]["attention_mask"].squeeze()  # squeeze to get rid of the batch dimension
        target_mask = self.targets[index]["attention_mask"].squeeze()  # convert [batch,dim] to [dim]

        # Label extraction - used for loss calculation from entry to entry
        labels = copy.deepcopy(target_ids)
        labels[labels == 0] = -100

        # Note: labels are used for loss calculation, target_ids are used for teacher forcing
        # What's teacher forcing you ask? Model calculates token-by-token as well, not only entry-by-entry
        return {"source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": target_mask,
                "labels": labels}

    # This function iterates over the dataset, extracts inputs and targets and tokenizes them
    def _build(self):
        for rownum, val in tqdm(self.data.iterrows()):  # Iterating over the dataframe
            passage, target = val[self.passage_column], val[self.question]

            input_ = f"Read this passage and generate an open question based on it: {passage}"  # Prompt engineering
            target = f"question: {str(target)}"

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=self.max_len_input, padding='max_length',
                truncation=True, return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len_output, padding='max_length',
                truncation=True,
                return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)
