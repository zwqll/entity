# -*- coding: utf-8 -*-
import re

if __name__ == '__main__':
	ENTITY_TYPES = ['CAR','OPT']

	with open('test_ner.txt','r') as f_truth, open('../aaa.txt','r') as f_infer:
		cnt_total = 0
		cnt_error = 0
		cnt_correct = 0
		truth_sentence_list = []
		infer_sentence_list = []
		temp_truth_sentence_list = []
		temp_infer_sentence_list = []
		for i,j in zip(f_truth.readlines(),f_infer.readlines()):
			if i == '\n':
				truth_sentence_list.append(temp_truth_sentence_list)
				infer_sentence_list.append(temp_infer_sentence_list)
				#cnt_total += 1
				temp_truth_sentence_list = []
				temp_infer_sentence_list = []
			else:
				temp_truth_sentence_list.append(i)
				temp_infer_sentence_list.append(j)
		
		for i,j in zip(truth_sentence_list,infer_sentence_list):
			temp_truth_entity_list = []
			for item_i in i:
				if len(item_i.strip().split('\t')) >= 3 and item_i.strip().split('\t')[2] == 'S-'+ENTITY_TYPES[1]:
					temp_truth_entity_list.append(item_i.strip().split('\t')[0])
			cnt_total += len(temp_truth_entity_list)
			for item_j in j:
				if len(item_j.strip().split('\t')) >= 3 and item_j.strip().split('\t')[2] == 'S-'+ENTITY_TYPES[1] and item_j.strip().split('\t')[0] in temp_truth_entity_list:
					cnt_correct += 1

		print 'precision:', 1.0*(cnt_correct)/cnt_total