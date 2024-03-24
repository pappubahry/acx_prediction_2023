import numpy as np

# Set penalise_empty to True for the Metaculus scoring in which a Peer score
# of zero is given for a non-answer.  Set to False for the original ACX
# results, in which a participant's final score is the mean of the Peer scores
# for the questions that they answered.
penalise_empty = True

true_answers = [1,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,0,1,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,1]

N_questions = len(true_answers)

participant_answers = []
participant_scores = []
participant_peer_scores = []
participant_peer_scores_by_question = []
mean_scores       = [0 for _ in range(N_questions)]
n_answers         = [0 for _ in range(N_questions)]

superforecasters = []

with open("2023blindmode_predictions_nodemo.csv", "r") as f:
	# Read answers
	for i, line in enumerate(f):
		if i == 0:
			continue
		cells = line.strip().split(",")
		answers = cells[4:]

		if cells[2] == "Yes":
			superforecasters.append(i - 1)

		participant_answers.append([])

		for j, ans in enumerate(answers):
			if ans.isdigit():
				p = int(ans) / 100
				participant_answers[-1].append(p)
				n_answers[j] += 1
			else:
				participant_answers[-1].append(np.nan)
	
	N_participants = len(participant_answers)

	# Compute log scores and average log scores
	for j in range(N_questions):
		for i in range(N_participants):
			participant_scores.append([])
			p = participant_answers[i][j]
			if np.isnan(p):
				participant_scores[i].append(np.nan)
			else:
				score = np.log(p) if true_answers[j] else np.log(1 - p)
				participant_scores[i].append(score)
				mean_scores[j] += score

		mean_scores[j] /= n_answers[j]
	
	# Compute peer scores
	N_excluded = 0
	for i in range(N_participants):
		participant_peer_scores_by_question.append([])
		participant_peer_scores.append(0)

		N_answered = 0

		for j in range(N_questions):
			factor = n_answers[j] / (n_answers[j] - 1)
			L = participant_scores[i][j]

			if np.isnan(L):
				participant_peer_scores_by_question[i].append(np.nan)
			else:
				peer_score = factor * (L - mean_scores[j])
				participant_peer_scores_by_question[i].append(peer_score)
				participant_peer_scores[i] += peer_score
				N_answered += 1

		if N_answered < N_questions / 4:
			# Don't count participants for the overall rankings if they
			# answered less than a quarter of the questions.
			N_excluded += 1
			participant_peer_scores[i] = -99999
		else:
			if penalise_empty:
				participant_peer_scores[i] /= N_questions
			else:
				participant_peer_scores[i] /= N_answered

max_score = max(participant_peer_scores)
i_max = participant_peer_scores.index(max_score)
print(f"Winner index (zero-indexed): {i_max}")
print()

# Scott is 2030
indices_to_print = [2030, i_max]

sorted_indices = np.argsort(participant_peer_scores)

for i in indices_to_print:
	line = f"Participant {i:4d}"
	line += f", score {participant_peer_scores[i]:.3f}"
	line += f", percentile {100*(np.sum(participant_peer_scores < participant_peer_scores[i]) - N_excluded) / (N_participants - N_excluded):.2f}"
	line += f", position: {N_participants - np.where(sorted_indices == i)[0][0]:4d} / {N_participants - N_excluded}"
	print(line)

print()

super_ranks = []
for i in superforecasters:
	super_ranks.append(N_participants - np.where(sorted_indices == i)[0][0])
print(f"Median superforecaster rank: {np.quantile(super_ranks, 0.5):.1f}")
print()

for j in range(N_questions):
	line = f"{j+1:2d}"
	for i in indices_to_print:
		line += f", {100*participant_answers[i][j]:2.0f}, {participant_peer_scores_by_question[i][j]:+.2f}"

	print(line)
