# coding:utf-8

import os
import re
import json
import time
import argparse
import subprocess


def get_args_parser():
	parser = argparse.ArgumentParser('extract dreamplace results', add_help=False)
	parser.add_argument('--results_path', default="./log/mms_adaptec1.log")
	parser.add_argument("--HPWL_savepath", default="./HPWL_results/baseline_results.json")
	parser.add_argument("--benchmark", default="mms")
	parser.add_argument("--root_path", default="./")
	parser.add_argument("--custom", action="store_true")
	parser.add_argument("--compare", action="store_true")
	parser.add_argument("--compare_path1", default="./HPWL_results/mms_baseline.json")
	parser.add_argument("--compare_path2", default="./HPWL_results/mms_selfopt_v2.json")
	
	return parser


def extract_all_wHPWL_values(content=None, log_file=None):
	wHPWL_values = []
	if content is not None:
		# for line in content:
		#     # print(line)
		#     match = re.search(r'wHPWL\s+([0-9.E+-]+)', line)
		#     if match:
		#         wHPWL_value = match.group(1)
		#         wHPWL_values.append(float(wHPWL_value))
		pattern = re.compile(r"wHPWL (\d+\.\d+E[+-]\d+), time")
		matches = pattern.findall(content)
		
		# The last value in matches is the final wHPWL value before 'time'
		if matches:
			wHPWL_values.append(matches[-1])  # Return the last matched wHPWL value
	
	elif content is None and log_file is not None:
		with open(log_file, 'r') as f:
			for line in f:
				match = re.search(r'wHPWL\s+([0-9.E+-]+)', line)
				if match:
					wHPWL_value = match.group(1)
					wHPWL_values.append(float(wHPWL_value))
	
	else:
		raise ValueError("Please input content or log file path!")
	
	if wHPWL_values:
		return wHPWL_values[-1]
	else:
		return None


def extract_iteration_value(content):
	"""
	Extracts the first iteration value associated with 'wHPWL' in the log data.

	Parameters:
	log_data (str): The log data as a string.

	Returns:
	str: The iteration value if found, otherwise None.
	"""
	# Regular expression to extract the iteration value just before 'wHPWL'
	pattern = r"iteration\s+(\d+),\s+wHPWL"
	
	# Search for the first match in the string
	match = re.search(pattern, content)
	
	if match:
		# Return the first captured group (the iteration number)
		return match.group(1)
	else:
		# Return None if no match is found
		return None


def extract_HPWL():
	args = get_args_parser().parse_args()
	log_file = args.results_path
	HPWL_results = extract_all_wHPWL_values(log_file)
	
	return HPWL_results


def extract_mms_hpwl_results_ours(save_path, custom=False, root_path=None):
	benchmarks = ["adaptec1", "adaptec2", "adaptec3", "adaptec4", "adaptec5",
	              "bigblue1", "bigblue2", "bigblue3", "bigblue4", "newblue1",
	              "newblue2", "newblue3", "newblue4", "newblue5", "newblue6", "newblue7"]
	
	total_HPWL = {}
	for benchmark in benchmarks:
		
		# set to default setting at first
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
		
		best_configs = os.path.join(root_path, "total_best_configs", "mms", benchmark)
		filenames = [i for i in os.listdir(best_configs) if ".py" in i]
		macro_filename = [i for i in filenames if "Macro" in i]
		preconditioner_filename = [i for i in filenames if "Preconditioner" in i]
		optimizer_filename = [i for i in filenames if "Optimizer" in i]
		
		if len(macro_filename) > 0:
			macro_filepath = os.path.join(best_configs, macro_filename[0])
			print("macro_filepath", macro_filepath)
			command_run = "cp {} dreamplace/BasicPlace.py".format(macro_filepath)
			process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
			                           stderr=subprocess.PIPE,
			                           universal_newlines=True, bufsize=1024 * 1024 * 5)
			results, stderr = process.communicate()  # results: output; stderr: error message
		else:
			pass
		
		if len(preconditioner_filename) > 0:
			preconditioner_filepath = os.path.join(best_configs, preconditioner_filename[0])
			command_run = "cp {} dreamplace/PlaceObj.py".format(preconditioner_filepath)
			process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
			                           stderr=subprocess.PIPE,
			                           universal_newlines=True, bufsize=1024 * 1024 * 5)
			results, stderr = process.communicate()  # results: output; stderr: error message
		else:
			pass
		
		if len(optimizer_filename) > 0:
			optimizer_filepath = os.path.join(best_configs, optimizer_filename[0])
			command_run = "cp {} dreamplace/CustomOptimizer.py".format(optimizer_filepath)
			process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
			                           stderr=subprocess.PIPE,
			                           universal_newlines=True, bufsize=1024 * 1024 * 5)
			results, stderr = process.communicate()  # results: output; stderr: error message
		else:
			pass
		
		if custom:
			command_run = "python dreamplace/Placer.py test/mms_llm4placement/{}.json".format(benchmark)
		else:
			command_run = "python dreamplace/Placer.py test/mms/{}.json".format(benchmark)
		print("command_run", command_run)
		start_time = time.time()
		try:
			process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
			                           stderr=subprocess.PIPE,
			                           universal_newlines=True, bufsize=1024 * 1024 * 5)
			results, stderr = process.communicate()  # results: output; stderr: error message
			cur_HPWL = extract_all_wHPWL_values(results)
			cur_iteration = extract_iteration_value(results)
			end_time = time.time()
			total_HPWL[benchmark] = {}
			total_HPWL[benchmark]["cur_HPWL"] = float(cur_HPWL)
			total_HPWL[benchmark]["cur_iteration"] = int(cur_iteration)
			total_HPWL[benchmark]["running-time"] = str(end_time - start_time)
		except Exception as e:
			print(e)
			total_HPWL[benchmark] = None
		print("benchmark", benchmark)
		print("cur_HPWL", cur_HPWL)
		print("cur_iteration", cur_iteration)
		print("running-time: {}".format(end_time - start_time))
	
	with open(save_path, "w")as json_f:
		json.dump(total_HPWL, json_f)


def extract_mms_hpwl_results(save_path, custom=False, root_path=None):
	benchmarks = ["adaptec1", "adaptec2", "adaptec3", "adaptec4", "adaptec5",
	              "bigblue1", "bigblue2", "bigblue3", "bigblue4", "newblue1",
	              "newblue2", "newblue3", "newblue4", "newblue5", "newblue6", "newblue7"]
	
	total_HPWL = {}
	for benchmark in benchmarks:
		if custom:
			command_run = "python dreamplace/Placer.py test/mms_llm4placement/{}.json".format(benchmark)
		else:
			command_run = "python dreamplace/Placer.py test/mms/{}.json".format(benchmark)
		print("command_run", command_run)
		start_time = time.time()
		try:
			process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
			                           stderr=subprocess.PIPE,
			                           universal_newlines=True, bufsize=1024 * 1024 * 5)
			results, stderr = process.communicate()  # results: output; stderr: error message
			cur_HPWL = extract_all_wHPWL_values(results)
			cur_iteration = extract_iteration_value(results)
			end_time = time.time()
			total_HPWL[benchmark] = {}
			total_HPWL[benchmark]["cur_HPWL"] = float(cur_HPWL)
			total_HPWL[benchmark]["cur_iteration"] = int(cur_iteration)
			total_HPWL[benchmark]["running-time"] = str(end_time - start_time)
		except Exception as e:
			print(e)
			total_HPWL[benchmark] = None
		print("benchmark", benchmark)
		print("cur_HPWL", cur_HPWL)
		print("cur_iteration", cur_iteration)
		print("running-time: {}".format(end_time - start_time))
	
	with open(save_path, "w")as json_f:
		json.dump(total_HPWL, json_f)


def extract_ispd2005free_hpwl_results_ours(save_path, custom=None, root_path=None):
	benchmarks = ["adaptec1", "adaptec2", "adaptec3", "adaptec4",
	              "bigblue1", "bigblue2", "bigblue3", "bigblue4"
	              ]
	
	total_HPWL = {}
	for benchmark in benchmarks:
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
		
		best_configs = os.path.join(root_path, "total_best_configs", "ispd2005free", benchmark + "_allfree")
		filenames = [i for i in os.listdir(best_configs) if ".py" in i]
		macro_filename = [i for i in filenames if "Macro" in i]
		preconditioner_filename = [i for i in filenames if "Preconditioner" in i]
		optimizer_filename = [i for i in filenames if "Optimizer" in i]
		
		if len(macro_filename) > 0:
			macro_filepath = os.path.join(best_configs, macro_filename[0])
			print("macro_filepath", macro_filepath)
			command_run = "cp {} dreamplace/BasicPlace.py".format(macro_filepath)
			process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
			                           stderr=subprocess.PIPE,
			                           universal_newlines=True, bufsize=1024 * 1024 * 5)
			results, stderr = process.communicate()  # results: output; stderr: error message
		else:
			pass
		
		if len(preconditioner_filename) > 0:
			preconditioner_filepath = os.path.join(best_configs, preconditioner_filename[0])
			command_run = "cp {} dreamplace/PlaceObj.py".format(preconditioner_filepath)
			process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
			                           stderr=subprocess.PIPE,
			                           universal_newlines=True, bufsize=1024 * 1024 * 5)
			results, stderr = process.communicate()  # results: output; stderr: error message
		else:
			pass
		
		if len(optimizer_filename) > 0:
			optimizer_filepath = os.path.join(best_configs, optimizer_filename[0])
			command_run = "cp {} dreamplace/CustomOptimizer.py".format(optimizer_filepath)
			process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
			                           stderr=subprocess.PIPE,
			                           universal_newlines=True, bufsize=1024 * 1024 * 5)
			results, stderr = process.communicate()  # results: output; stderr: error message
		else:
			pass
		
		if custom:
			command_run = "python dreamplace/Placer.py test/ispd2005free_llm4placement/{}_allfree.json".format(
				benchmark)
		else:
			command_run = "python dreamplace/Placer.py test/ispd2005free/{}_allfree.json".format(benchmark)
		print("command_run", command_run)
		start_time = time.time()
		try:
			process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
			                           stderr=subprocess.PIPE,
			                           universal_newlines=True, bufsize=1024 * 1024 * 5)
			results, stderr = process.communicate()  # results: output; stderr: error message
			cur_HPWL = extract_all_wHPWL_values(results)
			cur_iteration = extract_iteration_value(results)
			end_time = time.time()
			total_HPWL[benchmark] = {}
			total_HPWL[benchmark]["cur_HPWL"] = float(cur_HPWL)
			total_HPWL[benchmark]["cur_iteration"] = int(cur_iteration)
			total_HPWL[benchmark]["running-time"] = str(end_time - start_time)
		except Exception as e:
			print(e)
			total_HPWL[benchmark] = None
		print("benchmark", benchmark)
		print("cur_HPWL", cur_HPWL)
		print("cur_iteration", cur_iteration)
		print("running-time: {}".format(end_time - start_time))
	
	with open(save_path, "w")as json_f:
		json.dump(total_HPWL, json_f)


def extract_ispd2005free_hpwl_results(save_path, custom=None):
	benchmarks = ["adaptec1", "adaptec2", "adaptec3", "adaptec4",
	              "bigblue1", "bigblue2", "bigblue3", "bigblue4"
	              ]
	
	total_HPWL = {}
	for benchmark in benchmarks:
		if custom:
			command_run = "python dreamplace/Placer.py test/ispd2005free_llm4placement/{}_allfree.json".format(
				benchmark)
		else:
			command_run = "python dreamplace/Placer.py test/ispd2005free/{}_allfree.json".format(benchmark)
		print("command_run", command_run)
		start_time = time.time()
		try:
			process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
			                           stderr=subprocess.PIPE,
			                           universal_newlines=True, bufsize=1024 * 1024 * 5)
			results, stderr = process.communicate()  # results: output; stderr: error message
			cur_HPWL = extract_all_wHPWL_values(results)
			cur_iteration = extract_iteration_value(results)
			end_time = time.time()
			total_HPWL[benchmark] = {}
			total_HPWL[benchmark]["cur_HPWL"] = float(cur_HPWL)
			total_HPWL[benchmark]["cur_iteration"] = int(cur_iteration)
			total_HPWL[benchmark]["running-time"] = str(end_time - start_time)
		except Exception as e:
			print(e)
			total_HPWL[benchmark] = None
		print("benchmark", benchmark)
		print("cur_HPWL", cur_HPWL)
		print("cur_iteration", cur_iteration)
		print("running-time: {}".format(end_time - start_time))
	
	with open(save_path, "w")as json_f:
		json.dump(total_HPWL, json_f)


def extract_TILOS_hpwl_results(save_path, custom=None):
	benchmarks = ["ariane133", "ariane136", "bsg_chip", "mempool_tile_wrap", "NV_NVDLA_partition_c"]
	
	total_HPWL = {}
	for benchmark in benchmarks:
		if custom:
			command_run = "python dreamplace/Placer.py test/TILOS_llm4placement/{}.json".format(
				benchmark)
		else:
			command_run = "python dreamplace/Placer.py test/TILOS/{}.json".format(benchmark)
		print("command_run", command_run)
		start_time = time.time()
		try:
			process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
			                           stderr=subprocess.PIPE,
			                           universal_newlines=True, bufsize=1024 * 1024 * 5)
			results, stderr = process.communicate()  # results: output; stderr: error message
			cur_HPWL = extract_all_wHPWL_values(results)
			cur_iteration = extract_iteration_value(results)
			end_time = time.time()
			total_HPWL[benchmark] = {}
			total_HPWL[benchmark]["cur_HPWL"] = float(cur_HPWL)
			total_HPWL[benchmark]["cur_iteration"] = int(cur_iteration)
			total_HPWL[benchmark]["running-time"] = str(end_time - start_time)
		except Exception as e:
			print(e)
			total_HPWL[benchmark] = None
		print("benchmark", benchmark)
		print("cur_HPWL", cur_HPWL)
		print("cur_iteration", cur_iteration)
		print("running-time: {}".format(end_time - start_time))
	
	with open(save_path, "w")as json_f:
		json.dump(total_HPWL, json_f)


def extract_ispd2019_hpwl_results_ours(save_path, custom=False, root_path=None):
	# note that we omit ispd2019_test5 due to fence region problem, as explained in original dreamplace github
	# https://github.com/limbo018/DREAMPlace/blob/master/test/ispd2019/lefdef/regression.py#L36C1-L37C1
	benchmarks = ["ispd19_test1", "ispd19_test2", "ispd19_test3", "ispd19_test4",
	              "ispd19_test6", "ispd19_test7", "ispd19_test8", "ispd19_test9", "ispd19_test10"]
	
	total_HPWL = {}
	for benchmark in benchmarks:
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
		
		best_configs = os.path.join(root_path, "total_best_configs", "ispd2019", benchmark)
		filenames = [i for i in os.listdir(best_configs) if ".py" in i]
		macro_filename = [i for i in filenames if "Macro" in i]
		preconditioner_filename = [i for i in filenames if "Preconditioner" in i]
		optimizer_filename = [i for i in filenames if "Optimizer" in i]
		
		if len(macro_filename) > 0:
			macro_filepath = os.path.join(best_configs, macro_filename[0])
			print("macro_filepath", macro_filepath)
			command_run = "cp {} dreamplace/BasicPlace.py".format(macro_filepath)
			process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
			                           stderr=subprocess.PIPE,
			                           universal_newlines=True, bufsize=1024 * 1024 * 5)
			results, stderr = process.communicate()  # results: output; stderr: error message
		else:
			pass
		
		if len(preconditioner_filename) > 0:
			preconditioner_filepath = os.path.join(best_configs, preconditioner_filename[0])
			command_run = "cp {} dreamplace/PlaceObj.py".format(preconditioner_filepath)
			process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
			                           stderr=subprocess.PIPE,
			                           universal_newlines=True, bufsize=1024 * 1024 * 5)
			results, stderr = process.communicate()  # results: output; stderr: error message
		else:
			pass
		
		if custom:
			command_run = "python dreamplace/Placer.py test/ispd2019_llm4placement/{}.json".format(benchmark)
		else:
			command_run = "python dreamplace/Placer.py test/ispd2019/{}.json".format(benchmark)
		print("command_run", command_run)
		start_time = time.time()
		try:
			process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
			                           stderr=subprocess.PIPE,
			                           universal_newlines=True, bufsize=1024 * 1024 * 5)
			results, stderr = process.communicate()  # results: output; stderr: error message
			cur_HPWL = extract_all_wHPWL_values(results)
			cur_iteration = extract_iteration_value(results)
			end_time = time.time()
			total_HPWL[benchmark] = {}
			total_HPWL[benchmark]["cur_HPWL"] = float(cur_HPWL)
			total_HPWL[benchmark]["cur_iteration"] = int(cur_iteration)
			total_HPWL[benchmark]["running-time"] = str(end_time - start_time)
		except Exception as e:
			print(e)
			total_HPWL[benchmark] = None
		print("benchmark", benchmark)
		print("cur_HPWL", cur_HPWL)
		print("cur_iteration", cur_iteration)
		print("running-time: {}".format(end_time - start_time))
	
	with open(save_path, "w")as json_f:
		json.dump(total_HPWL, json_f)


def extract_ispd2019_hpwl_results(save_path, custom=False, root_path=None):
	# note that we omit ispd2019_test5 due to fence region problem, as explained in original dreamplace github
	# https://github.com/limbo018/DREAMPlace/blob/master/test/ispd2019/lefdef/regression.py#L36C1-L37C1
	benchmarks = ["ispd19_test1", "ispd19_test2", "ispd19_test3", "ispd19_test4",
	              "ispd19_test6", "ispd19_test7", "ispd19_test8", "ispd19_test9", "ispd19_test10"]
	
	total_HPWL = {}
	for benchmark in benchmarks:
		if custom:
			command_run = "python dreamplace/Placer.py test/ispd2019_llm4placement/{}.json".format(benchmark)
		else:
			command_run = "python dreamplace/Placer.py test/ispd2019/{}.json".format(benchmark)
		print("command_run", command_run)
		start_time = time.time()
		try:
			process = subprocess.Popen(command_run, shell=True, stdout=subprocess.PIPE,
			                           stderr=subprocess.PIPE,
			                           universal_newlines=True, bufsize=1024 * 1024 * 5)
			results, stderr = process.communicate()  # results: output; stderr: error message
			cur_HPWL = extract_all_wHPWL_values(results)
			cur_iteration = extract_iteration_value(results)
			end_time = time.time()
			total_HPWL[benchmark] = {}
			total_HPWL[benchmark]["cur_HPWL"] = float(cur_HPWL)
			total_HPWL[benchmark]["cur_iteration"] = int(cur_iteration)
			total_HPWL[benchmark]["running-time"] = str(end_time - start_time)
		except Exception as e:
			print(e)
			total_HPWL[benchmark] = None
		print("benchmark", benchmark)
		print("cur_HPWL", cur_HPWL)
		print("cur_iteration", cur_iteration)
		print("running-time: {}".format(end_time - start_time))
	
	with open(save_path, "w")as json_f:
		json.dump(total_HPWL, json_f)


if __name__ == "__main__":
	args = get_args_parser().parse_args()
	start_time = time.time()
	if args.benchmark == "mms":
		extract_mms_hpwl_results(save_path=args.HPWL_savepath, custom=args.custom)
	elif args.benchmark == "mms_ours":
		extract_mms_hpwl_results_ours(save_path=args.HPWL_savepath, custom=args.custom,
		                              root_path=args.root_path)
	elif args.benchmark == "ispd2005free":
		extract_ispd2005free_hpwl_results(save_path=args.HPWL_savepath, custom=args.custom)
	elif args.benchmark == "ispd2005free_ours":
		extract_ispd2005free_hpwl_results_ours(save_path=args.HPWL_savepath, custom=args.custom,
		                                       root_path=args.root_path)
	# elif args.benchmark == "TILOS":
	# 	extract_TILOS_hpwl_results(save_path=args.HPWL_savepath, custom=args.custom)
	elif args.benchmark == "ispd2019":
		extract_ispd2019_hpwl_results(save_path=args.HPWL_savepath, custom=args.custom)
	elif args.benchmark == "ispd2019_ours":
		extract_ispd2019_hpwl_results_ours(save_path=args.HPWL_savepath, custom=args.custom,
		                                   root_path=args.root_path)
	
	else:
		raise ValueError("Not valid benchmarks")
	end_time = time.time()
	print("total running time is: {}".format(end_time - start_time))
	
	# if args.compare:
	# 	with open(args.compare_path1, "r")as json_f:
	# 		compare_1 = json.load(json_f)
	# 	with open(args.compare_path2, "r")as json_f:
	# 		compare_2 = json.load(json_f)
	#
	# 	total_results, _total = {}, []
	# 	for case_key in list(compare_1.keys()):
	# 		improve_rate = 1. - float(compare_2[case_key]["cur_HPWL"] / compare_1[case_key]["cur_HPWL"])
	# 		total_results[case_key] = improve_rate
	# 		_total.append(improve_rate)
	# 		print("improve rate of case {} is: {}".format(case_key, str(improve_rate)))
	#
	# 	average_improve_rate = sum(_total) / len(_total)
	# 	print("average improve rate is:{}".format(average_improve_rate))