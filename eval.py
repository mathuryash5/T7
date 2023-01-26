# import json
# import os
# import sys

# from datasets import Dataset, DatasetDict
# from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer

# from main import read_json, add_ctr_text_and_normalize_data
# from nli_train import compute_metrics, preprocess_data

# if __name__ == "__main__":
# 	model_path = sys.argv[1]
# 	file_name = sys.argv[2]

# 	test_path = os.path.join("Complete_dataset", file_name)
# 	test_data = read_json(test_path)
# 	type = "test"
# 	for sampleid in test_data:
# 		if "Label" in test_data[sampleid]:
# 			type = "dev"
# 			break
# 		break
# 	normalized_test_data = add_ctr_text_and_normalize_data(test_data, type=type)
# 	actual_test_dataset = Dataset.from_list(normalized_test_data)
# 	if "label" in actual_test_dataset.features:
# 		print("Should log metrics")
# 		actual_test_dataset = actual_test_dataset.class_encode_column("label")
# 	tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
# 	model = AutoModelForSequenceClassification.from_pretrained(model_path)
# 	model.eval()
# 	encoded_actual_test_dataset = preprocess_data(actual_test_dataset, tokenizer)

# 	# arguments for Trainer
# 	test_args = TrainingArguments(
# 		output_dir=model_path,
# 		do_train=False,
# 		do_predict=True,
# 		per_device_eval_batch_size=16,
# 		dataloader_drop_last=False
# 	)

# 	# init trainer
# 	trainer = Trainer(
# 		model=model,
# 		args=test_args,
# 		tokenizer=tokenizer,
# 		compute_metrics=compute_metrics)

# 	predictions = trainer.predict(encoded_actual_test_dataset)
# 	print(predictions.metrics)
# 	predicted = predictions.predictions.argmax(axis=1)
# 	print(predicted[:10])
# 	print(predicted.shape)
# 	id2label = {0: "Contradiction", 1: "Entailment"}
# 	predicted_class = [id2label[x] for x in predicted]
# 	print(predicted_class)

# 	sampleids = []
# 	for sample in encoded_actual_test_dataset:
# 		sampleids.append(sample["sample_id"])

# 	res_dict = dict()
# 	for sampleid, pred in zip(sampleids, predicted_class):
# 		res_dict[sampleid] = {"Prediction": pred}

# 	print(res_dict)

# 	with open("predictions_{}.txt".format(file_name.split(".")[0]), "w+") as f:
# 		json.dump(res_dict, f)

import collections
import json
import os
import sys

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer

from main import read_json, add_ctr_text_and_normalize_data
from nli_train import compute_metrics, preprocess_data
from tqdm.auto import tqdm
import numpy as np
def preprocess_validation_examples(examples, tokenizer):
    # questions = [q.strip() for q in examples["question"]]

	max_length = 384  # The maximum length of a feature (question and context)
	doc_stride = 128  # The authorized overlap between two part of the context when splitting it is needed.
	tokenized_examples = tokenizer(examples["Premise"], examples["Statement"], padding='max_length', truncation="only_first",
        max_length=max_length,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        stride=doc_stride
        )
	# sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
	# offset_mapping = tokenized_examples.pop("offset_mapping")
	

	# inputs = tokenizer(
	# 	questions,
	# 	examples["context"],
	# 	max_length=max_length,
	# 	truncation="only_second",
	# 	stride=stride,
	# 	return_overflowing_tokens=True,
	# 	return_offsets_mapping=True,
	# 	padding="max_length",
	# )

	sample_map = tokenized_examples.pop("overflow_to_sample_mapping")
	example_ids = []

	for i in range(len(tokenized_examples["input_ids"])):
		sample_idx = sample_map[i]
		example_ids.append(examples["sample_id"][sample_idx])

		sequence_ids = tokenized_examples.sequence_ids(i)
		offset = tokenized_examples["offset_mapping"][i]
		tokenized_examples["offset_mapping"][i] = [
			o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
		]

	tokenized_examples["example_id"] = example_ids
	for key, values in examples.items():
		tokenized_examples[key] = [values[i] for i in sample_map]
	return tokenized_examples



def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size = 20, max_answer_length = 30):
    logits = raw_predictions
    print("Logits", logits)
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["sample_id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []
        
        # context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = logits[feature_index]
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            valid_answers.append({"text": example_index, "score": start_logits[start_indexes[0]], "index":start_indexes[0]})
            # end_logits = logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
        #     offset_mapping = features[feature_index]["offset_mapping"]

        #     # Update minimum null prediction.
        #     cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
        #     feature_null_score = start_logits[cls_index] #+ end_logits[cls_index]
        #     if min_null_score is None or min_null_score < feature_null_score:
        #         min_null_score = feature_null_score

        #     # Go through all possibilities for the `n_best_size` greater start and end logits.
        #     # end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
        #     for start_index in start_indexes:
        #         for end_index in end_indexes:
        #             # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
        #             # to part of the input_ids that are not in the context.
        #             if (
        #                 start_index >= len(offset_mapping)
        #                 or end_index >= len(offset_mapping)
        #                 or offset_mapping[start_index] is None
        #                 or offset_mapping[end_index] is None
        #             ):
        #                 continue
        #             # Don't consider answers with a length that is either < 0 or > max_answer_length.
        #             if end_index < start_index or end_index - start_index + 1 > max_answer_length:
        #                 continue

        #             start_char = offset_mapping[start_index][0]
        #             end_char = offset_mapping[end_index][1]
        #             valid_answers.append(
        #                 {
        #                     "score": start_logits[start_index] + end_logits[end_index],
        #                     "text": context[start_char: end_char]
        #                 }
        #             )
        
        if len(valid_answers) > 0:
             best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
             # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        #     # failure.
            print("This should not happen")
        predictions[example["sample_id"]] = (best_answer["score"], best_answer["index"])
        # # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        # if not squad_v2:
        #     predictions[example["id"]] = best_answer["text"]
        # else:
        #     answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
        #     predictions[example["id"]] = answer

    return predictions

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
	# encoded_actual_test_dataset = preprocess_validation_examples(actual_test_dataset, tokenizer)
	encoded_actual_test_dataset = actual_test_dataset.map(preprocess_validation_examples, batched=True, fn_kwargs={"tokenizer": tokenizer})
	# with torch.no_grad():
	# 	output = model(**encoded_actual_test_dataset)
	# print(output.keys())

	# arguments for Trainer
	test_args = TrainingArguments(
		output_dir=model_path,
		do_train=False,
		do_predict=True,
		per_device_eval_batch_size=16,
		dataloader_drop_last=False
	)
	
	# init trainer
	trainer = Trainer(
		model=model,
		args=test_args,
		tokenizer=tokenizer,
		compute_metrics=compute_metrics,
		eval_dataset=encoded_actual_test_dataset)
	
	for batch in trainer.get_eval_dataloader():
		break
	batch = {k: v.to(trainer.args.device) for k, v in batch.items()}
	with torch.no_grad():
		output = trainer.model(**batch)
	print(output.keys())
	
	predictions = trainer.predict(encoded_actual_test_dataset)
	encoded_actual_test_dataset.set_format(type=encoded_actual_test_dataset.format["type"], columns=list(encoded_actual_test_dataset.features.keys()))
	pred = postprocess_qa_predictions(actual_test_dataset, encoded_actual_test_dataset, predictions.predictions)
	print(pred)
	print(predictions.metrics)
	predicted = predictions.predictions.argmax(axis=1)
	print(predicted[:10])
	print(predicted.shape)
	id2label = {0: "Contradiction", 1: "Entailment"}
	predicted_class = [id2label[x] for x in predicted]
	print(predicted_class)
	new_predicted_class = [id2label[pred[x][-1]] for x in pred.keys()]
	# print("New predicted classes : ", new_predicted_class)
	sampleids = []
	for sample in encoded_actual_test_dataset:
		sampleids.append(sample["sample_id"])
	
	res_dict = dict()
	for sampleid, pred in zip(sampleids, new_predicted_class):
		res_dict[sampleid] = {"Prediction": pred}
	
	print(res_dict)
	
	with open("predictions_{}.txt".format(file_name.split(".")[0]), "w+") as f:
		json.dump(res_dict, f)
