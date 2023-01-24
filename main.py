import argparse
import json
import os

from datasets import Dataset
from transformers import AutoTokenizer

import constants
import nli_train


def read_json(filename):
	with open(filename) as json_file:
		data = json.load(json_file)
	return data


def get_ctr_section(ctrid, section):
	ctr_path = os.path.join(constants.CTR_DIR_PATH, ctrid + ".json")
	with open(ctr_path) as json_file:
		ctr = json.load(json_file)
	ctr_section_text = "".join(ctr[section])
	return ctr_section_text


def add_ctr_text_and_normalize_data(data, type="train"):
	res_data = []
	for id in data:
		ctr_text = "<PRIMARY> "
		sample = data[id]
		section = sample["Section_id"]
		if "Primary_id" in sample:
			primary_ctr_id = sample["Primary_id"]
			ctr_text += get_ctr_section(primary_ctr_id, section)
		if "Secondary_id" in sample:
			secondary_ctr_id = sample["Secondary_id"]
			ctr_text += " <SECONDARY> " + get_ctr_section(secondary_ctr_id, section)
		else:
			sample["Secondary_evidence_index"] = []
			sample["Secondary_id"] = "N/A"
		sample["Premise"] = ctr_text
		if type != "test":
			sample["label"] = sample.get("Label")
		sample["sample_id"] = id
		res_data.append(sample)
	return res_data


def parse_args():
	ap = argparse.ArgumentParser("arguments for nli training")
	ap.add_argument('-b', '--batch_size', type=int, default=8, help='batch size')
	ap.add_argument('-ep', '--epoch_num', type=int, default=1, help='epoch num')
	ap.add_argument('--fp16', type=int, default=0,
					help='use apex mixed precision training (1) or not (0); do not use this together with checkpoint')
	ap.add_argument('--check_point', '-cp', type=int, default=0,
					help='use checkpoint (1) or not (0); this is required for training bert-large or larger models; do not use '
						 'this together with apex fp16')
	ap.add_argument('--gpu', type=int, default=1, help='use gpu (1) or not (0)')
	ap.add_argument('-ss', '--scheduler_setting', type=str, default='WarmupLinear',
					choices=['WarmupLinear', 'ConstantLR', 'WarmupConstant', 'WarmupCosine',
							 'WarmupCosineWithHardRestarts'])
	ap.add_argument('-tm', '--trained_model', type=str, default='None',
					help='path to the trained model; make sure the trained model is consistent with the model you want to train')
	ap.add_argument('-mg', '--max_grad_norm', type=float, default=1., help='maximum gradient norm')
	ap.add_argument('-wp', '--warmup_percent', type=float, default=0.2,
					help='how many percentage of steps are used for warmup')
	ap.add_argument('-bt', '--bert_type', type=str, default='bert-base',
					help='transformer (bert) pre-trained model you want to use')
	ap.add_argument('--hans', type=int, default=0, help='use hans data (1) or not (0)')
	ap.add_argument('-rl', '--reinit_layers', type=int, default=0, help='reinitialise the last N layers')
	ap.add_argument('-fl', '--freeze_layers', type=int, default=0,
					help='whether to freeze all but the lasat few layers (1) or not (0)')
	ap.add_argument('-lr', '--learning_rate', type=float, default=2e-5, help='learning rate 2e-5')

	args = ap.parse_args()
	return args.batch_size, args.epoch_num, args.fp16, args.check_point, args.gpu, args.scheduler_setting, \
		   args.max_grad_norm, args.warmup_percent, args.bert_type, args.trained_model, args.hans, args.reinit_layers, \
		   args.freeze_layers, args.learning_rate


def prepare_data_for_dual_encoder_model(data):
	de_data = []
	id2label = {0: "Contradiction", 1: "Entailment"}
	for sample in data:
		sample_data = []
		sample_data.append(sample["sample_id"])
		sample_data.append(sample["sample_id"])
		sample_data.append(sample["Statement"])
		sample_data.append(sample["Premise"])
		sample_data.append(sample["label"])
		sample_data.append()




if __name__ == "__main__":
	batch_size, epoch_num, fp16, checkpoint, gpu, scheduler_setting, max_grad_norm, warmup_percent, bert_type, \
	trained_model, hans, reinit_layers, freeze_layers, learning_rate = parse_args()

	train_path = constants.TRAIN_PATH
	dev_path = constants.DEV_PATH
	test_path = constants.TEST_PATH

	train_data = read_json(train_path)
	dev_data = read_json(dev_path)
	test_data = read_json(test_path)

	print("Number of training examples: {}".format(len(train_data)))
	print("Number of dev examples: {}".format(len(dev_data)))
	print("Number of test examples: {}".format(len(test_data)))

	normalized_train_data = add_ctr_text_and_normalize_data(train_data)
	normalized_dev_data = add_ctr_text_and_normalize_data(dev_data)
	normalized_test_data = add_ctr_text_and_normalize_data(test_data, type="test")

	tokenizer = AutoTokenizer.from_pretrained(constants.MODEl_CHECKPOINT, use_fast=True)

	train_dataset = Dataset.from_list(normalized_train_data)
	train_dataset = train_dataset.class_encode_column("label")
	dev_dataset = Dataset.from_list(normalized_dev_data)
	dev_dataset = dev_dataset.class_encode_column("label")
	actual_test_dataset = Dataset.from_list(normalized_test_data)
	encoded_actual_test_dataset = nli_train.preprocess_data(actual_test_dataset, tokenizer)

	dataset = train_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
	dataset["dev"] = dataset["test"]
	dataset["test"] = dev_dataset
	print(dataset)

	encoded_dataset = nli_train.preprocess_data(dataset, tokenizer)
	trainer = nli_train.train(encoded_dataset, tokenizer, epochs=epoch_num, batch_size=batch_size, lr=learning_rate)

	trainer.train()
	trainer.evaluate()
	predictions = trainer.predict(encoded_dataset["test"])
	# print(predictions)
	predictions = trainer.predict(encoded_actual_test_dataset)
	predicted = predictions.predictions.argmax(axis=1)
	print(predicted[:10])
	print(predicted.shape)
	id2label = {0: "Contradiction", 1: "Entailment"}
	predicted_class = [id2label[x] for x in predicted]
	# print(predicted_class)

	sampleids = []
	for sample in encoded_actual_test_dataset:
		sampleids.append(sample["sample_id"])

	res_dict = dict()
	for sampleid, pred in zip(sampleids, predicted_class):
		res_dict[sampleid] = {"Prediction": pred}
	print(res_dict)
	with open("predictions.json", "w+") as f:
		json.dump(res_dict, f)
