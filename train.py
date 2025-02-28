from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset

# Load the pre-trained InLegalBERT (or a similar model) from Hugging Face
model_checkpoint = "law-ai/InLegalBERT"  # Replace with the correct model path if available
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

# Load your custom legal Q&A dataset
# Assume the dataset has fields: question, context, and answer.
dataset = load_dataset("json", data_files={"train": "train_data.json", "validation": "val_data.json"})

def preprocess_function(examples):
    # Tokenize questions and contexts together
    inputs = tokenizer(examples["question"], examples["context"], truncation=True, padding="max_length", max_length=512)
    # You would also compute the start and end positions of the answer in the context here.
    # For simplicity, we assume these positions are precomputed in your dataset.
    inputs["start_positions"] = examples["start_positions"]
    inputs["end_positions"] = examples["end_positions"]
    return inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Set training parameters
training_args = TrainingArguments(
    output_dir="./inlegal_qa_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained("./inlegal_qa_model")
tokenizer.save_pretrained("./inlegal_qa_model")
