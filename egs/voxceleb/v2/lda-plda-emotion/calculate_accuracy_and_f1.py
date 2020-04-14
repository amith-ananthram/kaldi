import csv

import numpy as np
from optparse import OptionParser
from collections import defaultdict

from sklearn.metrics import classification_report

def get_label(utterance):
	return utterance.split('-')[0]

def get_emotion_with_max_score(scores_by_emotion):
	max_score_by_emotion = {emotion: max(scores) \
		for emotion, scores in scores_by_emotion.items()}
	return list(sorted(lambda emotion_and_score: emotion_and_score[1], max_score_by_emotion.items()))[-1]

def get_emotion_with_max_avg_score(scores_by_emotion):
	avg_score_by_emotion = {emotion: np.mean(scores) \
		for emotion, scores in scores_by_emotion.items()}
	return list(sorted(lambda emotion_and_score: emotion_and_score[1], max_score_by_emotion.items()))[-1]

def main():
	usage = "usage: %prog [options]"
	parser = OptionParser(usage=usage)
	parser.add_option(
		'-f',
		'--score-file',
		dest='score_file'
	)
	(options, args) = parser.parse_args()

	test_scores = defaultdict(lambda: defaultdict(list))
	with open(options.score_file, 'r') as f:
		for row in csv.reader(f, delimite=' '):
			train_utterance, test_utterance, score = row
			test_scores[test_utterance][get_label(train_utterance)].append(score)

	real_labels = []
	max_score_labels = []
	max_avg_score_labels = []
	for test_utterance, scores_by_emotion in test_scores.items():
		if 'test' not in test_utterance:
			raise Exception('Are you sure %s is a test utterance?' % (test_utterance))

		real_labels.append(get_label(test_utterance))
		max_score_labels.append(get_emotion_with_max_score(scores_by_emotion))
		max_avg_score_labels.append(get_emotion_with_max_avg_score(scores_by_emotion))

	print(options.score_file)
	print('Classification using max score:')
	print(classification_report(real_labels, max_score_labels))
	print('Classification using max avg score:')
	print(classification_report(real_labels, max_avg_score_labels))

if __name__ == "__main__":
	main()