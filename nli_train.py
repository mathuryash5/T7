import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback

from constants import *


def preprocess_data(dataset, tokenizer):
	encoded_dataset = dataset.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
	return encoded_dataset


def preprocess_function(examples, tokenizer):
	max_length = 384  # The maximum length of a feature (question and context)
	doc_stride = 128  # The authorized overlap between two part of the context when splitting it is needed.
	tokenized_examples = tokenizer(examples["Premise"], examples["Statement"], padding='max_length', truncation="only_first",
			  max_length=max_length,
			  return_overflowing_tokens=True,
			  return_offsets_mapping=True,
			  stride=doc_stride
			  )
	sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
	offset_mapping = tokenized_examples.pop("offset_mapping")
	for key, values in examples.items():
		tokenized_examples[key] = [values[i] for i in sample_mapping]
	return tokenized_examples



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
	return f1_metric.compute(predictions=predictions, references=labels)


def train(encoded_dataset, tokenizer, epochs=1, batch_size=8, lr=2e-5):
	num_labels = 2
	id2label = {0: "Contradiction", 1: "Entailment"}
	label2id = {"Contradiction": 0, "Entailment": 1}
	model = AutoModelForSequenceClassification.from_pretrained(MODEl_CHECKPOINT, num_labels=num_labels,
															   id2label=id2label,
															   label2id=label2id)

	model.resize_token_embeddings(len(tokenizer))
	metric_name = "f1"
	model_name = MODEl_CHECKPOINT.split("/")[-1]

	args = TrainingArguments(
		f"{model_name}-finetuned-{TASK}",
		evaluation_strategy="steps",
		eval_steps=850,
		save_steps=850,
		save_strategy="steps",
		learning_rate=lr,
		per_device_train_batch_size=batch_size,
		per_device_eval_batch_size=batch_size,
		num_train_epochs=epochs,
		weight_decay=0.01,
		save_total_limit=2,
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
		compute_metrics=compute_metrics,
		callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
	)
	return trainer
