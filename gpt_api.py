# coding: utf-8

# import urllib3
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import openai  # version: v0.27.6
import traceback

DEBUG = False

# openai.log = "debug"
openai.api_key =   # your api-key
openai.api_base =  # your api-key url

openai.proxy = {
   # Your proxy here
}


def gpt_api_no_stream(messages: list, model='o1-preview'):
	try:
		completion = openai.ChatCompletion.create(
			# model="gpt-3.5-turbo",
			# model="gpt-4",
			model=model,
			messages=messages,
			timeout=30
		)
		# print(completion)
		
		msg = None
		choices = completion.get('choices', None)
		if choices:
			msg = choices[0]['message']['content']
		else:
			msg = completion['message']['content']
		return (True, msg)
	except Exception as err:
		return (False, f'OpenAI API: {err}')


def gpt_api_stream(messages: list, model='gpt-4'):
	completion = {'role': '', 'content': ''}
	try:
		response = openai.ChatCompletion.create(
			model=model,
			messages=messages,
			stream=True,
			# max_tokens=7000,
			# temperature=0.5,
			# presence_penalty=0,
		
		)
		for event in response:
			if event['choices'][0]['finish_reason'] == 'stop':
				if DEBUG:
					pass
				break
			for delta_k, delta_v in event['choices'][0]['delta'].items():
				if DEBUG:
					print(f'streaming: {delta_k} = {delta_v}')
				completion[delta_k] += delta_v
		messages.append(completion)
		msg = completion['content']
		return (True, msg)
	except Exception as err:
		if DEBUG:
			print(f"{traceback.format_exc()}")
		return (False, f'OpenAI API exception: {err} {completion}')
	

def gpt_text2embedding(prompt, model='text-embedding-3-large'):
	try:
		response = openai.Embedding.create(
			model=model,
			input=prompt
		)
		embeddings = response['data'][0]['embedding']
		return embeddings
	except Exception as err:
		return (False, f'OpenAI API : {err}')


if __name__ == '__main__':
	
	count = 20
	for _ in range(count):
		prompt = 'There are 9 birds in the tree, the hunter shoots one, how many birds are left in the tree？'  # api test
		print("prompt：", prompt)
		print("apikey：", openai.api_key)
		messages = [{'role': 'user', 'content': prompt}, ]
		# ret, msg = gpt_api_stream(messages, model='gpt-4-0125-preview')
		# ret, msg = gpt_api_no_stream(messages, model='gpt-4-0125-preview')
		# ret, msg = gpt_api_no_stream(messages, model='claude-3-sonnet-20240229')
		# ret, msg = gpt_api_no_stream(messages, model='claude-3-haiku-20240307')
		ret, msg = gpt_api_no_stream(messages, model='gpt-4o')
		embedding = gpt_text2embedding(prompt, model="text-embedding-3-small")
		print("embedding", len(embedding))
		print("msd：", msg)
		if DEBUG:
			print("messages：", messages)
		break



