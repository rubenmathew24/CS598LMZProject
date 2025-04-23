############################################################
# This file is to grab and normalize data from StackOverflow
############################################################


import os
from bs4 import BeautifulSoup
import requests
from googlesearch import search
from tqdm import tqdm
from termcolor import colored
import urllib

# Finds the domain of a URL
# Example: "https://stackoverflow.com/questions...." => "stackoverflow" 
def find_domain(url:str) -> str:
	domain = urllib.parse.urlparse(url).netloc
	domain_parts = domain.split(".")
	domain = domain_parts[1] if len(domain_parts) == 3 else domain_parts[0]
	return domain


def grab_stackoverflow_urls(topic:str, num_results=50, max_results=50):
	urls = []

	# Grab all StackOverflow URLs in the first 50 results from google
	try:
		for url in tqdm(search(topic, stop=num_results, pause=2), desc=f"Getting top {num_results} results", total=num_results, colour="YELLOW", leave=False):
			domain = find_domain(url)
			if domain == "stackoverflow":
				urls.append(url)

			# Lets you set max amount of results to gather for early break
			if len(urls) >= max_results: break
	except Exception as e:
		print(colored(f"Error: {e}", "red"))

	# Give Warning if less than 3 URLs are found
	if len(urls) < 3:
		print(colored("\nWarning: Less than 3 StackOverflow URLs found. Potentially Bad Query\n", "yellow"))

	return urls

def get_page_content(url:str, folder, number):
	# Get Page content
	try:
		r = requests.get(url)
		page = BeautifulSoup(r.text, features="html.parser")

		# Get Title
		title = page.find("title").get_text()
		title = title.removeprefix("python -")
		title = title.removesuffix("- Stack Overflow")
		title = title.strip()

		# Get Question Data

		# Grab block with question
		question_block = page.find("div", class_="question js-question")
		question_block = question_block.find("div", class_="s-prose js-post-body")

		# Parse tags of question into list
		question = []
		for child in question_block.find_all(recursive=False):
			if child.name == 'p':
				question.append(child.get_text())
			elif child.name == 'pre':
				question.append("```\n" + child.get_text() + "```")

		# Get Answer Data

		# Try for user accepted answer, then top answer
		answer_block = page.find("div", class_="answer js-answer accepted-answer js-accepted-answer")
		if answer_block == None: 
			answer_block = page.find("div", class_="answer js-answer")
		
		# If no answer found terminate early
		if answer_block == None:
			print(colored(f"\nError: No answer found for {url}", "red"))
			return
		
		# Grab block with answer
		answer_block = answer_block.find("div", class_="s-prose js-post-body")

		# Parse tags of answer into list
		answer = []
		for child in answer_block.find_all(recursive=False):
			if child.name == 'p':
				answer.append(child.get_text())
			elif child.name == 'pre':
				answer.append("```\n" + child.get_text() + "```")
		

		# Print to file
		with open(f"{folder}/{number}.log", "w") as f:

			print("Title:", title, "\n\n", file = f)

			print("Question:", file=f)
			print(*question, sep="\n", file = f)
			print("\n\n", file=f)

			print("Answer:", file=f)
			print(*answer, sep="\n", file=f)


	except Exception as e:
		print(colored(f"\nError: {e}", "red"))
	
	return


topics = {
	"django__django-14787": "AttributeError: 'functools.partial' object has no attribute '__name__'",
	"test_bad_query": "I love blueberries",
	"astropy__astropy-14182": "TypeError: RST.__init__() got an unexpected keyword argument 'header_rows'"
}


if __name__ == "__main__":

	LOG_FOLDER = "test_logs"

	if not os.path.exists(LOG_FOLDER):
		os.mkdir(LOG_FOLDER)

	for topic_id, topic in tqdm(topics.items(), desc="Topics Completed", colour="green", total=len(topics)):
		urls = grab_stackoverflow_urls(topic)
		# print(*urls, sep="\n")

		topic_folder = f"{LOG_FOLDER}/{topic_id}"

		if not os.path.exists(topic_folder):
			os.mkdir(topic_folder)

		for url_num, url in enumerate(urls):
			get_page_content(url, folder=topic_folder, number=url_num)