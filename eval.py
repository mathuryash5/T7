import json
import os
import sys

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer

from main import read_json, add_ctr_text_and_normalize_data
from nli_train import compute_metrics, preprocess_data

if __name__ == "__main__":
	model_path = sys.argv[1]
	file_name = sys.argv[2]

	test_path = os.path.join("Complete_dataset", file_name)
	test_data = read_json(test_path)
	type = "test"
	for sampleid in test_data:
		if "Label" in test_data[sampleid]:
			type = "dev"
			break
		break
	normalized_test_data = add_ctr_text_and_normalize_data(test_data, type=type)
	actual_test_dataset = Dataset.from_list(normalized_test_data)
	if "label" in actual_test_dataset.features:
		print("Should log metrics")
		actual_test_dataset = actual_test_dataset.class_encode_column("label")
	tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
	# tokenizer = AutoTokenizer.from_pretrained(constants.MODEl_CHECKPOINT, use_fast=True)
	special_tokens_dict = {'additional_special_tokens': ["[PRIMARY]", "[SECONDARY"]}
	num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
	model = AutoModelForSequenceClassification.from_pretrained(model_path)
	model.resize_token_embeddings(len(tokenizer))
	model.eval()

	# for batch in trainer.get_eval_dataloader():
	# 	break
	# batch = {k: v.to(trainer.args.device) for k, v in batch.items()}


	encoded_actual_test_dataset = preprocess_data(actual_test_dataset, tokenizer)

	with torch.no_grad():
		output = model(**encoded_actual_test_dataset)
	print(output.keys())

	# arguments for Trainer
	# test_args = TrainingArguments(
	# 	output_dir=model_path,
	# 	do_train=False,
	# 	do_predict=True,
	# 	per_device_eval_batch_size=16,
	# 	dataloader_drop_last=False
	# )
	#
	# # init trainer
	# trainer = Trainer(
	# 	model=model,
	# 	args=test_args,
	# 	tokenizer=tokenizer,
	# 	compute_metrics=compute_metrics)
	#
	# predictions = trainer.predict(encoded_actual_test_dataset)
	# print(predictions.metrics)
	# predicted = predictions.predictions.argmax(axis=1)
	# print(predicted[:10])
	# print(predicted.shape)
	# id2label = {0: "Contradiction", 1: "Entailment"}
	# predicted_class = [id2label[x] for x in predicted]
	# print(predicted_class)
	#
	# sampleids = []
	# for sample in encoded_actual_test_dataset:
	# 	sampleids.append(sample["sample_id"])
	#
	# res_dict = dict()
	# for sampleid, pred in zip(sampleids, predicted_class):
	# 	res_dict[sampleid] = {"Prediction": pred}
	#
	# print(res_dict)
	#
	# with open("predictions_{}.txt".format(file_name.split(".")[0]), "w+") as f:
	# 	json.dump(res_dict, f)
