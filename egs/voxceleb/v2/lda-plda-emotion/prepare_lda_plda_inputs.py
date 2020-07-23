import sys
from kaldiio import ReadHelper, WriteHelper

from collections import defaultdict

CREMA_D_PATH = 'nnet-emotion/cremad/outputs/data/all_cremad/'
MELD_PATH = 'nnet-emotion/meld/outputs/data/all_meld/'
IEMOCAP_PATH = 'nnet-emotion/iemocap/all_iemocap/'

VALID_TRAINING_CORPORA = set([
	'cremad',
	'meld',
	*['iemocap%s' % i for i in range(1, 6)]
])

EMOTION_TO_ID = {emotion: id for id, emotion in 
	enumerate(['anger/disgust', 'fear/surprise', 'happiness', 'neutral', 'sadness'])}

# example utterance id: anger/disgust-1001_DFA_ANG_XX
def get_cremad_utterances(speech_dir, text_dir):
	utterances = defaultdict(dict)
	with open('%s/utt2spk' % (CREMA_D_PATH), 'r') as f:
		for line in f.readlines():
			utterance_id, _ = line.split(' ')

			if utterance_id in utterances:
				raise Exception("Duplicate utterance: %s" % utterance_id)

			utterances[utterance_id]['emotion'] = utterance_id.split('-')[0]

	with ReadHelper('scp:%s/cremad/xvector.scp' % speech_dir):
		for utterance_id, speech_vector in reader:
			if utterance_id in utterances:
				raise Exception("Speech vector for unknown utterance: %s" % utterance_id)

			utterances[utterance_id]['speech'] = speech_vector

	# get random dialog text vector for same speech

# example utterance id: anger/disgust-dev-1-11
def get_meld_utterances(speech_dir, text_dir):
	utterances = defaultdict(dict)
	with open('%s/utt2spk' % (MELD_PATH), 'r') as f:
		for line in f.readlines():
			utterance_id, _ = line.split(' ')

			if utterance_id in utterances:
				raise Exception("Duplicate utterance: %s" % utterance_id)

			utterances[utterance_id]['emotion'] = utterance_id.split('-')[0]

	with ReadHelper('scp:%s/meld/xvector.scp' % speech_dir):
		for utterance_id, speech_vector in reader:
			if utterance_id in utterances:
				raise Exception("Speech vector for unknown utterance: %s" % utterance_id)

			utterances[utterance_id]['speech'] = speech_vector

	# get text vector 

# example utterance id: sadness-test-05M-script-02_2-032-M
def get_iemocap_utterances(speech_dir, text_dir, subset):
	utterances = defaultdict(dict)
	with open('%s/utt2spk' % (IEMOCAP_PATH), 'r') as f:
		for line in f.readlines():
			utterance_id, _ = line.split(' ')
			session = int(utterance_id.split('-')[2][0:2])

			if session != int(subset[-1:]):
				continue

			if utterance_id in utterances:
				raise Exception("Duplicate utterance: %s" % utterance_id)

			utterances[utterance_id]['session'] = session
			utterances[utterance_id]['emotion'] = utterance_id.split('-')[0]

	with ReadHelper('scp:%s/meld/xvector.scp' % speech_dir):
		for utterance_id, speech_vector in reader:
			if utterance_id in utterances:
				raise Exception("Speech vector for unknown utterance: %s" % utterance_id)

			utterances[utterance_id]['speech'] = speech_vector

	# get text vectors

def get_corpus_utterances(speech_dir, text_dir, corpus):
	if corpus == 'cremad':
		return get_cremad_utterances(speech_dir, text_dir)
	elif corpus == 'meld':
		return get_meld_utterances(speech_dir, text_dir)
	else:
		return get_iemocap_utterances(speech_dir, text_dir, corpus)

def get_utterances(speech_dir, text_dir, corpora):
	utterances = {}
	for corpus in corpora:
		for utterance_id, utterance in get_corpus_utterances(speech_dir, text_dir, corpus).items():
			if utterance_id in utterances:
				raise Exception("Duplicate utterance: %s" % utterance_id)
			utterances[utterance_id] = utterance
	return utterances


def write_output_files(prefix, utterances, output_dir):
	# utt2spk -- index
	# spk2utt -- index
	# xvector.scp, xvector.ark

	utt2spk = {}
	spk2utt = defaultdict(list)
	for utterance_id, utterance in utterances.items():
		emotion = EMOTION_TO_ID[utterance['emotion']]

		utt2spk[utterance_id] = utterance
		spk2utt[utterance].append(utterance_id)

	with open('%s/%s_utt2spk' % (output_dir, prefix), 'w') as f:
		for utterance_id in sorted(utt2spk.keys()):
			f.write("%s %s\n" % (utterance_id, utt2spk[utterance_id]))

	with open('%s/%s_spk2utt' % (output_dir, prefix), 'w') as f:
		for speaker in sorted(spk2utt.keys()):
			f.write("%s %s\n" % (speaker, ' '.join(spk2utt[speaker])))

	with WriteHelper(
		'ark,scp:/%s/%s_xvector.ark,/%s/%s_xvector.scp' % (
			output_dir, prefix, output_dir, prefix)) as writer:
		for utterance_id in sorted(utterances.keys()):
			writer(
				utterance_id, 
				np.concatenate(
					utterances[utterance_id]['speech'],
					utterances[utterance_id]['text']
				)
			)

def write_trials_file(train_utterances, test_utterances, output_dir):
	with open('%s/trials', 'w') as f:
		for train_utterance_id in sorted(train_utterances.keys()):
			for test_utterance_id in sorted(test_utterances.keys()):
				if train_utterances[train_utterance_id] == test_utterances[test_utterance_id]:
					label = 'target'
				else:
					label = 'nontarget'
				f.write('%s %s %s\n' % (train_utterance_id, test_utterance_id, label))


speech_dir = sys.argv[1]
text_dir = sys.argv[2]
train_corpora = set(sys.argv[3].split(','))
output_dir = sys.argv[4]

##
# validate specified training corpora
##

invalid_corpora = train_corpora - VALID_TRAINING_CORPORA
if len(invalid_corpora) > 0:
	raise Exception("Unknown corpora: %s" % (invalid_corpora))

iemocap_subset_to_exclude = list(filter(lambda corpus: 'iemocap' in corpus, train_corpora))
if len(iemocap_subset_to_exclude) > 1:
	raise Exception("Can't specify excluding more than 1 IEMOCAP subset: %s" % (iemocap_subset_to_exclude))

test_corpora = ['iemocap%s' % i for i in range(1, 6)] \ 
	if len(iemocap_subset_to_exclude) == 0 else iemocap_subset_to_exclude

train_utterances = get_utterances(speech_dir, text_dir, train_corpora)
test_utterances = get_utterances(speech_dir, text_dir, test_corpora)

write_output_files('train', train_utterances, output_dir)
write_output_files('test', test_utterances, output_dir)
write_trials_file(train_utterances, test_utterances, output_dir)