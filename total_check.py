# coding: utf-8

'''
to ensure the stability, store each case performance
first calculate each case, timeout=500
second calculate improvement ratio in dictionary
finally store in dictionary,  with total improvement ratio
'''

import re
import os
import time
import json
import random
import argparse
import subprocess
from extract_results import extract_all_wHPWL_values

# Set CUDA_VISIBLE_DEVICES to 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def chunk_list(files, num_chunks):
	# First, sort the files list
	files.sort()
	
	# Divide the sorted list into num_chunks approximately equal chunks
	avg = len(files) / float(num_chunks)
	chunks = []
	last = 0.0
	
	while last < len(files):
		chunks.append(files[int(last):int(last + avg)])
		last += avg
	
	return chunks


def get_chunk(files, num_chunks, idx):
	# Get the list of chunks
	chunks = chunk_list(files, num_chunks)
	
	# Ensure the index is within the bounds
	if 0 <= idx < len(chunks):
		return chunks[idx]
	else:
		raise IndexError(f"Index {idx} is out of range. Must be between 0 and {len(chunks) - 1}.")


def total_macro_check(args):
	case_name = args.case_name
	benchmark = args.benchmark
	root_path = args.root_path
	timeout = args.timeout
	sel_idx = args.sel_idx
	
	if benchmark == "ispd2005free":
		total_case_name = ["adaptec1_allfree", "adaptec2_allfree", "adaptec3_allfree", "adaptec4_allfree",
		                   "bigblue1_allfree", "bigblue2_allfree", "bigblue3_allfree", "bigblue4_allfree"]
	elif benchmark == "mms":
		total_case_name = ["adaptec1", "adaptec2", "adaptec3", "adaptec4", "adaptec5",
		                   "bigblue1", "bigblue2", "bigblue3", "bigblue4", "newblue1",
		                   "newblue2", "newblue3", "newblue4", "newblue5", "newblue6", "newblue7"]
	elif benchmark == "ispd2019":
		total_case_name = ["ispd19_test1", "ispd19_test2", "ispd19_test3", "ispd19_test4", "ispd19_test5",
		                   "ispd19_test6", "ispd19_test7", "ispd19_test8", "ispd19_test9", "ispd19_test10"]
	else:
		raise ValueError("Please ensure your benchmark is correct.")
	
	total_macro_init_path = os.path.join(root_path, "total_macro_init")
	total_macro_init_filenames = [i for i in os.listdir(total_macro_init_path) if "best" in i]
	total_macro_init_filenames = sorted(total_macro_init_filenames)
	print("len(total_macro_init_filenames)", len(total_macro_init_filenames))
	
	saved_macro_init_directory = os.path.join(root_path, "total_best_configs/macro_init/{}".format(benchmark))
	saved_macro_init_total_filenames = [i for i in os.listdir(saved_macro_init_directory) if "MacroInit" in i]
	saved_best_macro_init_filenames = [i for i in saved_macro_init_total_filenames if "best" in i]
	saved_macro_init_filenames = [i for i in saved_macro_init_total_filenames if
	                              i not in saved_best_macro_init_filenames]
	saved_best_macro_init_filenames = ["best_MacroInit" + i.split("MacroInit")[1].replace(".json", "") for i in
	                                   saved_best_macro_init_filenames]
	saved_macro_init_filenames = ["best_" + i.replace(".json", "") for i in saved_macro_init_filenames]
	saved_macro_init_filenames.extend(saved_best_macro_init_filenames)
	
	total_macro_init_filenames = [i for i in total_macro_init_filenames if i not in saved_macro_init_filenames]
	print("after-len(total_macro_init_filenames)", len(total_macro_init_filenames))
	
	# selected_macro_init_filenames = total_macro_init_filenames  # no parallel
	selected_macro_init_filenames = get_chunk(total_macro_init_filenames, 16, sel_idx)
	print("selected_macro_init_filenames", selected_macro_init_filenames[:5])
	
	## default macro init
	command_run = "cp ./dreamplace/BasicPlace_backup.py ./dreamplace/BasicPlace.py"
	process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
	                           stderr=subprocess.PIPE,
	                           universal_newlines=True, bufsize=1024 * 1024 * 5)
	results, stderr = process.communicate()  # results: output; stderr: error message
	
	## default precondition
	command_run = "cp ./dreamplace/PlaceObj_backup.py ./dreamplace/PlaceObj.py"
	process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
	                           stderr=subprocess.PIPE,
	                           universal_newlines=True, bufsize=1024 * 1024 * 5)
	results, stderr = process.communicate()  # results: output; stderr: error message
	
	## default custom optimizer
	command_run = "cp ./dreamplace/CustomOptimizer_backup.py ./dreamplace/CustomOptimizer.py"
	process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
	                           stderr=subprocess.PIPE,
	                           universal_newlines=True, bufsize=1024 * 1024 * 5)
	results, stderr = process.communicate()  # results: output; stderr: error message
	
	######### run default performance baseline  #########
	save_path = "./prompt/{}_baseline.json".format(benchmark)
	if os.path.exists(save_path):
		pass
	else:
		total_HPWL = {}
		for case in total_case_name:
			command_run = "python dreamplace/Placer.py test/{}_llm4placement/{}.json".format(benchmark, case)
			print("command_run", command_run)
			start_time = time.time()
			try:
				process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
				                           stderr=subprocess.PIPE,
				                           universal_newlines=True, bufsize=1024 * 1024 * 5)
				results, stderr = process.communicate()  # results: output; stderr: error message
				cur_HPWL = extract_all_wHPWL_values(results)
				end_time = time.time()
				total_HPWL[case] = {}
				total_HPWL[case]["cur_HPWL"] = float(cur_HPWL)
				total_HPWL[case]["running-time"] = str(end_time - start_time)
			except Exception as e:
				print(e)
				total_HPWL[case] = None
			print("case", case)
			print("cur_HPWL", cur_HPWL)
			print("running-time: {}".format(end_time - start_time))
		
		with open(save_path, "w")as json_f:
			json.dump(total_HPWL, json_f)
	
	######### run macro init performance  #########
	with open(save_path, "r")as json_f:
		baseline = json.load(json_f)
	
	save_path_macro_init = os.path.join(root_path, "total_best_configs/macro_init/{}".format(benchmark))
	if os.path.exists(save_path_macro_init):
		pass
	else:
		os.makedirs(save_path_macro_init)
	
	for filename in selected_macro_init_filenames:
		try:
			cur_save_path = os.path.join(save_path_macro_init, filename.replace("best_", "") + ".json")
			macro_filename = os.path.join(total_macro_init_path, filename)
			# cp current macro file to dreamplace install directory
			command_run = "cp {} ./dreamplace/BasicPlace.py".format(macro_filename)
			process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
			                           stderr=subprocess.PIPE,
			                           universal_newlines=True, bufsize=1024 * 1024 * 5)
			results, stderr = process.communicate()  # results: output; stderr: error message
			
			total_HPWL = {}
			total_improvement = []
			for case_name in total_case_name:
				command_run = "python dreamplace/Placer.py test/{}_llm4placement/{}.json".format(benchmark, case_name)
				# timeout is important for each case!!
				start_time = time.time()
				try:
					process = subprocess.run(command_run, shell=True, text=True, capture_output=True, timeout=timeout)
					results, stderr = process.stdout, process.stderr
					cur_HPWL = extract_all_wHPWL_values(results)
					end_time = time.time()
					total_HPWL[case_name] = {}
					total_HPWL[case_name]["cur_HPWL"] = float(cur_HPWL)
					total_HPWL[case_name]["running-time"] = str(end_time - start_time)
					total_HPWL[case_name]["improve_ratio"] = 1 - float(cur_HPWL) / baseline[case_name]["cur_HPWL"] * 1.
					total_improvement.append(total_HPWL[case_name]["improve_ratio"])
				except Exception as e:
					print(e)
					total_HPWL[case_name]["cur_HPWL"] = 1E+16
					total_HPWL[case_name]["improve_ratio"] = 1 - float(cur_HPWL) / baseline[case_name]["cur_HPWL"] * 1.
					total_improvement.append(total_HPWL[case]["improve_ratio"])
				print("case", case_name)
				print("cur_HPWL", cur_HPWL)
				print("running-time: {}".format(end_time - start_time))
			total_HPWL["total_improvement_rato"] = sum(total_improvement) / len(total_improvement) * 1.
			total_HPWL["path"] = macro_filename
			
			with open(cur_save_path, "w")as json_f:
				json.dump(total_HPWL, json_f)
			
			if float(total_HPWL["total_improvement_rato"]) > 0:
				cur_best_save_path = os.path.join(save_path_macro_init,
				                                  "best_{}".format(str(total_HPWL["total_improvement_rato"])) +
				                                  filename.replace("best_", "") + ".json"
				                                  )
				
				with open(cur_best_save_path, "w")as json_f:
					json.dump(total_HPWL, json_f)
		except Exception as e:
			print(e)
			continue


def total_preconditioner_check(args):
	case_name = args.case_name
	benchmark = args.benchmark
	root_path = args.root_path
	timeout = args.timeout
	sel_idx = args.sel_idx
	chunk_num = args.chunk_list
	
	if benchmark == "ispd2005free":
		total_case_name = ["adaptec1_allfree", "adaptec2_allfree", "adaptec3_allfree", "adaptec4_allfree",
		                   "bigblue1_allfree", "bigblue2_allfree", "bigblue3_allfree", "bigblue4_allfree"]
	elif benchmark == "mms":
		total_case_name = ["adaptec1", "adaptec2", "adaptec3", "adaptec4", "adaptec5",
		                   "bigblue1", "bigblue2", "bigblue3", "bigblue4", "newblue1",
		                   "newblue2", "newblue3", "newblue4", "newblue5", "newblue6", "newblue7"]
	elif benchmark == "ispd2019":
		total_case_name = ["ispd19_test1", "ispd19_test2", "ispd19_test3", "ispd19_test4", "ispd19_test5",
		                   "ispd19_test6", "ispd19_test7", "ispd19_test8", "ispd19_test9", "ispd19_test10"]
	else:
		raise ValueError("Please ensure your benchmark is correct.")
	
	total_preconditioner_init_path = os.path.join(root_path, "total_preconditioner_init")
	total_preconditioner_init_filenames = [i for i in os.listdir(total_preconditioner_init_path) if ".py" in i]
	total_preconditioner_init_filenames = sorted(total_preconditioner_init_filenames)
	print("len(total_preconditioner_init_filenames)", len(total_preconditioner_init_filenames))
	
	saved_preconditioner_init_directory = os.path.join(root_path,
	                                                   "total_best_configs/preconditioner_init/{}".format(benchmark))
	# check if existing savepath
	if os.path.exists(saved_preconditioner_init_directory):
		pass
	else:
		os.mkdir(saved_preconditioner_init_directory)
	
	saved_preconditioner_init_total_filenames = [i for i in os.listdir(saved_preconditioner_init_directory) if
	                                             "preconditionerInit" in i]
	saved_best_preconditioner_init_filenames = [i for i in saved_preconditioner_init_total_filenames if "best" in i]
	saved_preconditioner_init_filenames = [i for i in saved_preconditioner_init_total_filenames if
	                                       i not in saved_best_preconditioner_init_filenames]
	saved_best_preconditioner_init_filenames = [
		"best_preconditionerInit" + i.split("preconditionerInit")[1].replace(".json", "") for i in
		saved_best_preconditioner_init_filenames]
	saved_preconditioner_init_filenames = ["best_" + i.replace(".json", "") for i in
	                                       saved_preconditioner_init_filenames]
	saved_preconditioner_init_filenames.extend(saved_best_preconditioner_init_filenames)
	
	total_preconditioner_init_filenames = [i for i in total_preconditioner_init_filenames if
	                                       i not in saved_preconditioner_init_filenames]
	print("after-len(total_preconditioner_init_filenames)", len(total_preconditioner_init_filenames))
	
	# selected_preconditioner_init_filenames = total_preconditioner_init_filenames  # no parallel
	selected_preconditioner_init_filenames = get_chunk(total_preconditioner_init_filenames, chunk_num, sel_idx)
	print("selected_preconditioner_init_filenames", selected_preconditioner_init_filenames[:5])
	
	## default macro init
	command_run = "cp ./dreamplace/BasicPlace_backup.py ./dreamplace/BasicPlace.py"
	process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
	                           stderr=subprocess.PIPE,
	                           universal_newlines=True, bufsize=1024 * 1024 * 5)
	results, stderr = process.communicate()  # results: output; stderr: error message
	
	## default precondition
	command_run = "cp ./dreamplace/PlaceObj_backup.py ./dreamplace/PlaceObj.py"
	process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
	                           stderr=subprocess.PIPE,
	                           universal_newlines=True, bufsize=1024 * 1024 * 5)
	results, stderr = process.communicate()  # results: output; stderr: error message
	
	## default custom optimizer
	command_run = "cp ./dreamplace/CustomOptimizer_backup.py ./dreamplace/CustomOptimizer.py"
	process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
	                           stderr=subprocess.PIPE,
	                           universal_newlines=True, bufsize=1024 * 1024 * 5)
	results, stderr = process.communicate()  # results: output; stderr: error message
	
	######### run default performance baseline  #########
	save_path = "./prompt/{}_baseline.json".format(benchmark)
	if os.path.exists(save_path):
		pass
	else:
		total_HPWL = {}
		for case in total_case_name:
			###### select best macro ######
			macro_init_path = os.path.join(root_path, "total_best_configs/{}/{}".format(benchmark, case))
			macro_filename = [i for i in os.listdir(macro_init_path) if "MacroInit" in i]
			if len(macro_filename) > 1:
				raise ValueError("should be only one initialization algorithm")
			elif len(macro_filename) == 1:
				macro_filename = os.path.join(macro_init_path, macro_filename[0])
				print("macro_filename", macro_filename)
				## evolved init algorithm
				command_run = "cp {} ./dreamplace/BasicPlace.py".format(macro_filename)
				process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
				                           stderr=subprocess.PIPE,
				                           universal_newlines=True, bufsize=1024 * 1024 * 5)
				results, stderr = process.communicate()  # results: output; stderr: error message
			else:
				pass  # no suitable macro init file
			
			command_run = "python dreamplace/Placer.py test/{}_llm4placement/{}.json".format(benchmark, case)
			print("command_run", command_run)
			start_time = time.time()
			try:
				process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
				                           stderr=subprocess.PIPE,
				                           universal_newlines=True, bufsize=1024 * 1024 * 5)
				results, stderr = process.communicate()  # results: output; stderr: error message
				cur_HPWL = extract_all_wHPWL_values(results)
				end_time = time.time()
				total_HPWL[case] = {}
				total_HPWL[case]["cur_HPWL"] = float(cur_HPWL)
				total_HPWL[case]["running-time"] = str(end_time - start_time)
			except Exception as e:
				print(e)
				total_HPWL[case] = None
			print("case", case)
			print("cur_HPWL", cur_HPWL)
			print("running-time: {}".format(end_time - start_time))
		
		with open(save_path, "w")as json_f:
			json.dump(total_HPWL, json_f)
	
	######### run preconditioner init performance  #########
	with open(save_path, "r")as json_f:
		baseline = json.load(json_f)
	
	save_path_preconditioner_init = os.path.join(root_path, "total_best_configs/preconditioner_init/{}".format(benchmark))
	if os.path.exists(save_path_preconditioner_init):
		pass
	else:
		os.makedirs(save_path_preconditioner_init)
	
	for filename in selected_preconditioner_init_filenames:
		print("filename", filename)
		# try:
		cur_save_path = os.path.join(save_path_preconditioner_init, filename.replace("best_", "") + ".json")
		preconditioner_filename = os.path.join(total_preconditioner_init_path, filename)
		# cp current preconditioner file to dreamplace install directory
		command_run = "cp {} ./dreamplace/BasicPlace.py".format(preconditioner_filename)
		process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
		                           stderr=subprocess.PIPE,
		                           universal_newlines=True, bufsize=1024 * 1024 * 5)
		results, stderr = process.communicate()  # results: output; stderr: error message
		
		total_HPWL = {}
		total_improvement = []
		for case_name in total_case_name:
			###### select best macro ######
			macro_init_path = os.path.join(root_path, "total_best_configs/{}/{}".format(benchmark, case_name))
			macro_filename = [i for i in os.listdir(macro_init_path) if "MacroInit" in i]
			if len(macro_filename) > 1:
				raise ValueError("should be only one initialization algorithm")
			elif len(macro_filename) == 1:
				macro_filename = os.path.join(macro_init_path, macro_filename[0])
				print("macro_filename", macro_filename)
				## evolved init algorithm
				command_run = "cp {} ./dreamplace/BasicPlace.py".format(macro_filename)
				process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
				                           stderr=subprocess.PIPE,
				                           universal_newlines=True, bufsize=1024 * 1024 * 5)
				results, stderr = process.communicate()  # results: output; stderr: error message
			else:
				pass  # no suitable macro init file
			
			command_run = "python dreamplace/Placer.py test/{}_llm4placement/{}.json".format(benchmark, case_name)
			# timeout is important for each case!!
			start_time = time.time()
			# try:
			process = subprocess.run(command_run, shell=True, text=True, capture_output=True, timeout=timeout)
			results, stderr = process.stdout, process.stderr
			print("results", results)
			cur_HPWL = extract_all_wHPWL_values(results)
			print("cur_HPWL", cur_HPWL)
			end_time = time.time()
			total_HPWL[case_name] = {}
			total_HPWL[case_name]["cur_HPWL"] = float(cur_HPWL)
			total_HPWL[case_name]["running-time"] = str(end_time - start_time)
			print("case_name", case_name)
			print("baseline", baseline)
			total_HPWL[case_name]["improve_ratio"] = 1 - float(cur_HPWL) / baseline[case_name]["cur_HPWL"] * 1.
			total_improvement.append(total_HPWL[case_name]["improve_ratio"])
			# except Exception as e:
			# 	print(e)
			# 	total_HPWL[case_name]["cur_HPWL"] = 1E+16
			# 	total_HPWL[case_name]["improve_ratio"] = 1 - float(cur_HPWL) / baseline[case_name]["cur_HPWL"] * 1.
			# 	total_improvement.append(total_HPWL[case]["improve_ratio"])
			print("case", case_name)
			print("cur_HPWL", cur_HPWL)
			print("running-time: {}".format(end_time - start_time))
		total_HPWL["total_improvement_rato"] = sum(total_improvement) / len(total_improvement) * 1.
		total_HPWL["path"] = preconditioner_filename
		
		with open(cur_save_path, "w")as json_f:
			json.dump(total_HPWL, json_f)
		
		if float(total_HPWL["total_improvement_rato"]) > 0:
			cur_best_save_path = os.path.join(save_path_preconditioner_init,
			                                  "best_{}".format(str(total_HPWL["total_improvement_rato"])) +
			                                  filename.replace("best_", "") + ".json"
			                                  )
			
			with open(cur_best_save_path, "w")as json_f:
				json.dump(total_HPWL, json_f)


def total_optimizer_check(args):
	case_name = args.case_name
	benchmark = args.benchmark
	root_path = args.root_path
	timeout = args.timeout
	sel_idx = args.sel_idx
	
	if benchmark == "ispd2005free":
		total_case_name = ["adaptec1_allfree", "adaptec2_allfree", "adaptec3_allfree", "adaptec4_allfree",
		                   "bigblue1_allfree", "bigblue2_allfree", "bigblue3_allfree", "bigblue4_allfree"]
	elif benchmark == "mms":
		total_case_name = ["adaptec1", "adaptec2", "adaptec3", "adaptec4", "adaptec5",
		                   "bigblue1", "bigblue2", "bigblue3", "bigblue4", "newblue1",
		                   "newblue2", "newblue3", "newblue4", "newblue5", "newblue6", "newblue7"]
	elif benchmark == "ispd2019":
		total_case_name = ["ispd19_test1", "ispd19_test2", "ispd19_test3", "ispd19_test4", "ispd19_test5",
		                   "ispd19_test6", "ispd19_test7", "ispd19_test8", "ispd19_test9", "ispd19_test10"]
	else:
		raise ValueError("Please ensure your benchmark is correct.")
	
	total_macro_init_path = os.path.join(root_path, "total_macro_init")
	total_macro_init_filenames = [i for i in os.listdir(total_macro_init_path) if "best" in i]
	total_macro_init_filenames = sorted(total_macro_init_filenames)
	print("len(total_macro_init_filenames)", len(total_macro_init_filenames))
	
	saved_macro_init_directory = os.path.join(root_path, "total_best_configs/macro_init/{}".format(benchmark))
	saved_macro_init_total_filenames = [i for i in os.listdir(saved_macro_init_directory) if "MacroInit" in i]
	saved_best_macro_init_filenames = [i for i in saved_macro_init_total_filenames if "best" in i]
	saved_macro_init_filenames = [i for i in saved_macro_init_total_filenames if
	                              i not in saved_best_macro_init_filenames]
	saved_best_macro_init_filenames = ["best_MacroInit" + i.split("MacroInit")[1].replace(".json", "") for i in
	                                   saved_best_macro_init_filenames]
	saved_macro_init_filenames = ["best_" + i.replace(".json", "") for i in saved_macro_init_filenames]
	saved_macro_init_filenames.extend(saved_best_macro_init_filenames)
	
	total_macro_init_filenames = [i for i in total_macro_init_filenames if i not in saved_macro_init_filenames]
	print("after-len(total_macro_init_filenames)", len(total_macro_init_filenames))
	
	# selected_macro_init_filenames = total_macro_init_filenames  # no parallel
	selected_macro_init_filenames = get_chunk(total_macro_init_filenames, 16, sel_idx)
	print("selected_macro_init_filenames", selected_macro_init_filenames[:5])
	
	## default macro init
	command_run = "cp ./dreamplace/BasicPlace_backup.py ./dreamplace/BasicPlace.py"
	process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
	                           stderr=subprocess.PIPE,
	                           universal_newlines=True, bufsize=1024 * 1024 * 5)
	results, stderr = process.communicate()  # results: output; stderr: error message
	
	## default precondition
	command_run = "cp ./dreamplace/PlaceObj_backup.py ./dreamplace/PlaceObj.py"
	process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
	                           stderr=subprocess.PIPE,
	                           universal_newlines=True, bufsize=1024 * 1024 * 5)
	results, stderr = process.communicate()  # results: output; stderr: error message
	
	## default custom optimizer
	command_run = "cp ./dreamplace/CustomOptimizer_backup.py ./dreamplace/CustomOptimizer.py"
	process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
	                           stderr=subprocess.PIPE,
	                           universal_newlines=True, bufsize=1024 * 1024 * 5)
	results, stderr = process.communicate()  # results: output; stderr: error message
	
	######### run default performance baseline  #########
	save_path = "./prompt/{}_baseline.json".format(benchmark)
	if os.path.exists(save_path):
		pass
	else:
		total_HPWL = {}
		for case in total_case_name:
			command_run = "python dreamplace/Placer.py test/{}_llm4placement/{}.json".format(benchmark, case)
			print("command_run", command_run)
			start_time = time.time()
			try:
				process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
				                           stderr=subprocess.PIPE,
				                           universal_newlines=True, bufsize=1024 * 1024 * 5)
				results, stderr = process.communicate()  # results: output; stderr: error message
				cur_HPWL = extract_all_wHPWL_values(results)
				end_time = time.time()
				total_HPWL[case] = {}
				total_HPWL[case]["cur_HPWL"] = float(cur_HPWL)
				total_HPWL[case]["running-time"] = str(end_time - start_time)
			except Exception as e:
				print(e)
				total_HPWL[case] = None
			print("case", case)
			print("cur_HPWL", cur_HPWL)
			print("running-time: {}".format(end_time - start_time))
		
		with open(save_path, "w")as json_f:
			json.dump(total_HPWL, json_f)
	
	######### run macro init performance  #########
	with open(save_path, "r")as json_f:
		baseline = json.load(json_f)
	
	save_path_macro_init = os.path.join(root_path, "total_best_configs/macro_init/{}".format(benchmark))
	if os.path.exists(save_path_macro_init):
		pass
	else:
		os.makedirs(save_path_macro_init)
	
	for filename in selected_macro_init_filenames:
		try:
			cur_save_path = os.path.join(save_path_macro_init, filename.replace("best_", "") + ".json")
			macro_filename = os.path.join(total_macro_init_path, filename)
			# cp current macro file to dreamplace install directory
			command_run = "cp {} ./dreamplace/BasicPlace.py".format(macro_filename)
			process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
			                           stderr=subprocess.PIPE,
			                           universal_newlines=True, bufsize=1024 * 1024 * 5)
			results, stderr = process.communicate()  # results: output; stderr: error message
			
			total_HPWL = {}
			total_improvement = []
			for case_name in total_case_name:
				command_run = "python dreamplace/Placer.py test/{}_llm4placement/{}.json".format(benchmark, case_name)
				# timeout is important for each case!!
				start_time = time.time()
				try:
					process = subprocess.run(command_run, shell=True, text=True, capture_output=True, timeout=timeout)
					results, stderr = process.stdout, process.stderr
					cur_HPWL = extract_all_wHPWL_values(results)
					end_time = time.time()
					total_HPWL[case_name] = {}
					total_HPWL[case_name]["cur_HPWL"] = float(cur_HPWL)
					total_HPWL[case_name]["running-time"] = str(end_time - start_time)
					total_HPWL[case_name]["improve_ratio"] = 1 - float(cur_HPWL) / baseline[case_name]["cur_HPWL"] * 1.
					total_improvement.append(total_HPWL[case_name]["improve_ratio"])
				except Exception as e:
					print(e)
					total_HPWL[case_name]["cur_HPWL"] = 1E+16
					total_HPWL[case_name]["improve_ratio"] = 1 - float(cur_HPWL) / baseline[case_name]["cur_HPWL"] * 1.
					total_improvement.append(total_HPWL[case]["improve_ratio"])
				print("case", case_name)
				print("cur_HPWL", cur_HPWL)
				print("running-time: {}".format(end_time - start_time))
			total_HPWL["total_improvement_rato"] = sum(total_improvement) / len(total_improvement) * 1.
			total_HPWL["path"] = macro_filename
			
			with open(cur_save_path, "w")as json_f:
				json.dump(total_HPWL, json_f)
			
			if float(total_HPWL["total_improvement_rato"]) > 0:
				cur_best_save_path = os.path.join(save_path_macro_init,
				                                  "best_{}".format(str(total_HPWL["total_improvement_rato"])) +
				                                  filename.replace("best_", "") + ".json"
				                                  )
				
				with open(cur_best_save_path, "w")as json_f:
					json.dump(total_HPWL, json_f)
		except Exception as e:
			print(e)
			continue


if __name__ == "__main__":
	# Create the parser
	parser = argparse.ArgumentParser(description="A simple script demonstrating argparse usage.")
	parser.add_argument('--benchmark', default="mms",
	                    type=str, help='Name of the benchamrk to process')
	parser.add_argument('--case_name', default="adaptec1_allfree",
	                    type=str, help='Name of the case to process')
	parser.add_argument('--root_path', default="./DAC25",
	                    type=str, help='root path to save best configs')
	parser.add_argument('--timeout', default=600,
	                    type=int, help='timeout for running case, pay attention to timeout!')
	parser.add_argument('--chunk_list', default=16, type=int)
	parser.add_argument('--sel_idx', default=0,
	                    type=int, help='default 0-15')
	args = parser.parse_args()
	total_preconditioner_check(args)
