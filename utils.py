# coding: utf-8

import re
import os
import time
import random
import argparse
import subprocess


from gpt_api import gpt_api_no_stream  # another gpt4 account


def call_llm(prompt, model="gpt-4o"):
	messages = [{'role': 'user', 'content': prompt}, ]
	try:
		ret, msg = gpt_api_no_stream(messages, model=model)
	except Exception as e:
		msg = None
	
	return msg


def call_optimization_analysis(optimizer_analysis_prompt, match_pattern, trails=5):
	optimizer_analysis_response = None
	for idx in range(trails):  # at most trails times call for obtaining optimizer analysis
		try:
			print("This is the {} time to obtain optimizer analysis".format(str(idx + 1)))
			optimizer_analysis_response = call_llm(prompt=optimizer_analysis_prompt, model="gpt-4o")
			cur_optimizer_analysis = extract_optimizer_analysis(optimizer_analysis_response, match_pattern)
			if cur_optimizer_analysis is not None:
				break
			else:
				time.sleep(1)
		except Exception as e:
			pass
	
	return cur_optimizer_analysis


def call_generated_optimizer(cur_optimizer_generate_prompt, cur_best_optimizer):
	cur_generated_optimizer, key_improvement_points = None, None
	try:
		cur_optimizer_generate_prompt = cur_optimizer_generate_prompt.replace("@@cur-optimizer@@", cur_best_optimizer)
		optimizer_generated_response = call_llm(prompt=cur_optimizer_generate_prompt, model="gpt-4o")
		cur_generated_optimizer = extract_optimizer_code(optimizer_generated_response)
		key_improvement_points = extract_keypoints_summary(optimizer_generated_response)
	except Exception as e:
		cur_generated_optimizer, key_improvement_points = None, None
	
	return cur_generated_optimizer, key_improvement_points


def extract_optimizer_name(content):
	# Pattern to match class definition with the optimizer name
	pattern = r"class\s+(\w+)\s*\(BaseOptimizer\):"
	
	# Search for the pattern in the text
	match = re.search(pattern, content)
	
	if match:
		# Return the captured optimizer name
		return match.group(1)
	else:
		return None


def extract_optimizer_analysis(content, match_pattern="Given Optimizer Analysis"):
	# Regular expression to match Python code blocks
	pattern = r'```analysis\n(.*?)```'
	matches = re.findall(pattern, content, re.DOTALL)
	
	# Find the block that contains the optimizer class
	optimizer_analysis = None
	for match in matches:
		if match_pattern in match:
			optimizer_analysis = match
			break
	
	if optimizer_analysis:
		# Remove any leading or trailing whitespace
		optimizer_analysis = optimizer_analysis.strip()
		return optimizer_analysis
	else:
		return None


def extract_keypoints_summary(content, match_pattern):
	# Regular expression to match Python code blocks
	pattern = r"'''Key improvement points summary\n(.*?)'''"
	matches = re.findall(pattern, content, re.DOTALL)
	summary = None
	print(content)
	try:
		if matches:
			for match in matches:
				if match_pattern in match:
					summary = match
					break
	except Exception as e:
		return summary



def extract_optimizer_code(content):
	# Regular expression to match Python code blocks
	pattern = r'```python\n(.*?)```'
	matches = re.findall(pattern, content, re.DOTALL)
	
	# Find the block that contains the optimizer class
	optimizer_code = None
	for match in matches:
		if 'class CusOptimizer' in match:
			optimizer_code = match
			break
	
	if optimizer_code:
		# Remove any leading or trailing whitespace
		optimizer_code = optimizer_code.strip()
		return optimizer_code
	else:
		return None


def extract_macroinit_code(content):
	# Regular expression to match Python code blocks
	pattern = r'```python\n(.*?)```'
	matches = re.findall(pattern, content, re.DOTALL)
	
	# Find the block that contains the optimizer class
	optimizer_code = None
	for match in matches:
		if 'self.init_pos' in match:
			optimizer_code = match
			break
	
	if optimizer_code:
		# Remove any leading or trailing whitespace
		optimizer_code = optimizer_code.strip()
		return optimizer_code
	else:
		return None


def replace_macro_init_content(file_content, new_content):
	pattern = r"(###move-layout-start###)(.*?)(###move-layout-end###)"
	
	def replacement(match):
		start, _, end = match.groups()
		# Get the current indentation level
		lines = match.group(0).splitlines()
		indent = len(lines[1]) - len(lines[1].lstrip())
		# Indent the new content appropriately
		indented_new_content = "\n".join(" " * indent + line if line else line for line in new_content.splitlines())
		return f"{start}\n{indented_new_content}\n{' ' * indent}{end}"
	
	return re.sub(pattern, replacement, file_content, flags=re.DOTALL)


def extract_preconditioner_code(content):
	# Regular expression to match Python code blocks
	pattern = r'```python\n(.*?)```'
	matches = re.findall(pattern, content, re.DOTALL)
	
	# Find the block that contains the optimizer class
	optimizer_code = None
	for match in matches:
		if 'density_weight' in match:
			optimizer_code = match
			break
	
	if optimizer_code:
		# Remove any leading or trailing whitespace
		optimizer_code = optimizer_code.strip()
		return optimizer_code
	else:
		return None


def extract_ideas_content(content):
	pattern = r'```markdown\n(.*?)```'
	total_ideas = []
	try:
		matches = re.findall(pattern, content, re.DOTALL)[0]
		ideas = matches.split("@@@")
		total_ideas = [i.strip() for i in ideas if len(i) > 100]
	except Exception as e:
		pass
	
	return total_ideas


def extract_reference_content(content):
	pattern = r'@@@reference\n(.*?)@@@'
	# name_pattern = r'reference optimizer name:\s*\*\*(.*?)\*\*'
	try:
		reference = re.findall(pattern, content, re.DOTALL)[0]
	# reference_name = re.search(name_pattern, reference, re.DOTALL).group(1)
	except Exception as e:
		reference = None
	
	return reference


def replace_pre_condition_content(file_content, new_content):
	pattern = r"(###pre-condition-start###)(.*?)(###pre-condition-end###)"
	
	def replacement(match):
		start, _, end = match.groups()
		# Get the current indentation level
		lines = match.group(0).splitlines()
		indent = len(lines[1]) - len(lines[1].lstrip())
		# Indent the new content appropriately
		indented_new_content = "\n".join(" " * indent + line if line else line for line in new_content.splitlines())
		return f"{start}\n{indented_new_content}\n{' ' * indent}{end}"
	
	return re.sub(pattern, replacement, file_content, flags=re.DOTALL)


def save_best_optimizer(cur_generated_optimizer_savepath,
                        cur_generated_optimizer,
                        idx, cur_HPWL):
	cur_gen_optimizer_best_savepath = os.path.join(cur_generated_optimizer_savepath,
	                                               "Optimizer_best_{}_{}.py".format(str(idx), str(cur_HPWL)))
	with open(cur_gen_optimizer_best_savepath, "w") as py_f:
		py_f.write(cur_generated_optimizer)


def get_files_in_directory(directory_path):
	files = []
	# List all items in the directory
	with os.scandir(directory_path) as entries:
		for entry in entries:
			# Check if the entry is a file
			if entry.is_file():
				files.append(entry.name)
	return files


# Function to extract the best value from filename, with keyword filtering
def extract_best_value(filename, keyword=None):
	# Check if the filename contains the keyword if a keyword is provided
	if keyword and keyword.lower() not in filename.lower():
		return float('inf')  # Ignore the file if keyword doesn't match
	
	# Use regex to find the number before the '.py' extension
	match = re.search(r'_(\d+\.\d+)\.py$', filename)
	if match:
		return float(match.group(1))  # Convert the extracted value to a float
	return float('inf')  # Return a high value if no match is found


# Function to find the file with the lowest best value for a specific keyword
def find_lowest_best_file(file_names, keyword=None):
	# Use the keyword filter and find the filename with the lowest best value
	return min(file_names, key=lambda f: extract_best_value(f, keyword))

def extract_best_value(filenames):
	pattern = r'_(\d+\.\d+)\.py$'
	result = {filename: float(re.search(pattern, filename).group(1)) for filename in filenames}