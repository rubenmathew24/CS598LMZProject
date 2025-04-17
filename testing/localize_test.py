import re
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict

# Load the SWE-bench Lite dataset
dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

# Grab and clean up from log file
def get_prediction(target):
	filepath = f"results/swe-bench-lite/file_level/localization_logs/{target}.log"

	with open(filepath, "r", encoding="utf-8") as f:
		lines = f.readlines()

	# Reverse to scan from the end
	lines = [line.strip() for line in lines[::-1]]
	border = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - INFO - ```$")
	extracted_files = []

	for line in lines[1:]:
		if border.match(line):
			break
		
		extracted_files.append(line)

	# Reverse again to restore original order
	return extracted_files[::-1]

# Check if any file is in ground truth
def evaluate(target, row, files):

	# Extract all files modified in the patch using the diff header format
	diff_file_matches = re.findall(r"^diff --git a/(.*?) b/", row["patch"], flags=re.MULTILINE)

	for file in files:

		if file in diff_file_matches: return True

	return False

if __name__ == "__main__":

	total = len(dataset)
	matches = 0

	total_repos = defaultdict(int)
	match_repos = defaultdict(int)
	

	for example in tqdm(dataset, "Evaluating File Localization:"):
		target = example["instance_id"]
		repo = example["repo"]

		total_repos[repo] += 1
		files = get_prediction(target)

		if evaluate(target, example, files):
			matches += 1
			match_repos[repo] += 1


	print("File Localization:", f"{(matches/total)*100:.2f}%", f"({matches}/{total})\n")

	# Print per-repo breakdown
	print("Per-Repository Accuracy:")
	for repo in sorted(total_repos.keys()):
		correct = match_repos.get(repo, 0)
		total_r = total_repos[repo]
		accuracy = (correct / total_r) * 100
		print(f"- {repo}: {accuracy:.2f}% ({correct}/{total_r})")
