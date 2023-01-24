import evaluate
import numpy as np
from datasets import load_metric
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer

from constants import *


def preprocess_data(dataset, tokenizer):
	encoded_dataset = dataset.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
	return encoded_dataset


def preprocess_function(examples, tokenizer):
	return tokenizer(examples["Premise"], examples["Statement"], truncation=True, max_length=128)


def compute_metrics(eval_pred):
	accuracy_metric = evaluate.load("accuracy")
	precision_metric = evaluate.load("precision")
	f1_metric = evaluate.load("f1")
	recall_metric = evaluate.load('recall')
	predictions, labels = eval_pred
	predictions = np.argmax(predictions, axis=1)
	metric_dict = {}
	metric_dict["accuracy"] = accuracy_metric.compute(predictions=predictions, references=labels)
	metric_dict["precision"] = precision_metric.compute(predictions=predictions, references=labels)
	metric_dict["f1"] = f1_metric.compute(predictions=predictions, references=labels)
	metric_dict["recall"] = recall_metric.compute(predictions=predictions, references=labels)
	return f1_metric.compute(predictions=predictions, references=labels), \
		   recall_metric.compute(predictions=predictions, references=labels)


def train(encoded_dataset, tokenizer, epochs=1, batch_size=8, lr=2e-5):
	num_labels = 2
	id2label = {0: "Contradiction", 1: "Entailment"}
	label2id = {"Contradiction": 0, "Entailment": 1}
	model = AutoModelForSequenceClassification.from_pretrained(MODEl_CHECKPOINT, num_labels=num_labels,
															   id2label=id2label,
															   label2id=label2id)
	metric_name = "f1"
	model_name = MODEl_CHECKPOINT.split("/")[-1]

	args = TrainingArguments(
		f"{model_name}-finetuned-{TASK}",
		evaluation_strategy="epoch",
		save_strategy="epoch",
		learning_rate=lr,
		per_device_train_batch_size=batch_size,
		per_device_eval_batch_size=batch_size,
		num_train_epochs=epochs,
		weight_decay=0.01,
		save_total_limit=1,
		load_best_model_at_end=True,
		metric_for_best_model=metric_name,
		push_to_hub=False,
	)

	trainer = Trainer(
		model,
		args,
		train_dataset=encoded_dataset["train"],
		eval_dataset=encoded_dataset["dev"],
		tokenizer=tokenizer,
		compute_metrics=compute_metrics
	)
	return trainer
