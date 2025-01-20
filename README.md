# T5 Question Generation with Optuna

This project uses a fine-tuned T5 model for generating questions from context passages. It incorporates **Optuna** for hyperparameter optimization and **PyTorch Lightning** for training and evaluation.

## Features
- **Dataset**: Utilizes the SQuAD dataset, loading 50% and splitting it into training, validation, and test sets (80%/10%/10%).
- **Model**: Fine-tunes the T5-small model for question generation.
- **Hyperparameter Optimization**: Employs Optuna to optimize learning rate, batch size, and weight decay.
- **Evaluation**: Outputs validation loss during tuning and test results for the best model.
- **Example Predictions**: Saves a sample of model-generated questions with context and target questions.

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   
2. Install required dependencies:
    ```bash
    pip install -r requirements.txt

3. Install additional dependencies:
    ```bash
   pip install optuna-integration[pytorch_lightning]
   pip install sentencepiece

## How to Run

The files full_finetune_t5.py and baseline_t5.py contain the scripts
for training the models and saving them on your local machines. Due 
to github LFS constraints, we have not uploaded the models as they 
are rather large. That being said, we would love to demonstrate how
our model works so if you are grading this project and want to see
a demo it definitely can be arranged. Currently, this functionality
is in the try_model.py script.

## Project Limitations

The project was initially planned to be of a much 
bigger scale but due to time constraints we made many compromises to 
have a working version of the project in time for the deadlines.
The main idea was to have a QG - QA pipeline with BERT answering the 
generated question from T5. Another comparison we wanted to try out
was fine-tuning T5 using LoRA and seeing how the models compare against
each other in the task. Lastly, as is always the case with LLM fine-tuning
and training, we wanted to train on a bigger portion of the SQuAD dataset
and have a much wider variety of hyperparameter tuning search space. 
We resorted to cutting some corners in terms of testing out different 
parameters and things like extensive prompt engineering as finetuning
the model took quite a while. 




