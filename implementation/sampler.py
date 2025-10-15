# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for sampling new programs."""
from collections.abc import Collection, Sequence
import numpy as np
from vllm import SamplingParams
from vllm import LLM as vLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests
import os
from datetime import datetime
from google import genai
from funsearch.implementation import evaluator
from funsearch.implementation import programs_database

import ast
import textwrap
import re

def extract_between_dollars(text: str) -> str:
    """
    Extracts the substring between the first pair of $$ ... $$.
    Returns an empty string if not found.
    """
    start = text.find("$$")
    if start == -1:
        text = textwrap.dedent(text).strip()
        return text
    end = text.find("$$", start + 2)
    if end == -1:
        return text
    
    return text[start + 2:end].strip()

def extract_code_block(text):
    """Extracts the first Python code block enclosed in triple backticks."""
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    return match.group(1) if match else None


def is_reward_hacking_from_body(code_body: str) -> bool:
    code_wrapped = f"def dummy():\n{textwrap.indent(code_body, '    ')}"
    try:
        tree = ast.parse(code_wrapped)
        func_node = tree.body[0]  # the wrapped dummy function

        assigned_constants = {}
        modified_vars = set()
        env_step_found = False

        for node in ast.walk(func_node):
            # Detect assignment of constants
            if isinstance(node, ast.Assign):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    var_name = node.targets[0].id
                    if isinstance(node.value, (ast.Constant, ast.Num, ast.Str, ast.List, ast.Tuple)):
                        assigned_constants[var_name] = node.lineno

            # Detect augmented assignment or mutation
            elif isinstance(node, (ast.AugAssign, ast.Call)):
                targets = []
                if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                    targets = [node.target.id]
                elif isinstance(node, ast.Call):
                    # Check for env.step(...) call
                    if isinstance(node.func, ast.Attribute):
                        if (isinstance(node.func.value, ast.Name) and
                            node.func.value.id == "env" and
                            node.func.attr == "step"):
                            env_step_found = True

                    # Also track function call argument modifications
                    for arg in node.args:
                        if isinstance(arg, ast.Name):
                            targets.append(arg.id)
                for t in targets:
                    modified_vars.add(t)

        # If env.step() not found, it's reward hacking
        if not env_step_found:
            return True

        for node in ast.walk(func_node):
            if isinstance(node, ast.Return):
                if isinstance(node.value, (ast.Constant, ast.Num, ast.Str, ast.List, ast.Tuple)):
                    return True
                elif isinstance(node.value, ast.Name):
                    var_name = node.value.id
                    if var_name in assigned_constants and var_name not in modified_vars:
                        return True

        return False
    except Exception as e:
        print(f"Error parsing code: {e}")
        return False
        
class LLM:
  """Language model that predicts continuation of provided source code."""

  def __init__(self, samples_per_prompt: int, model=None, tokenizer=None, model_type='ollama') -> None:
    self._samples_per_prompt = samples_per_prompt
    self.tokenizer = tokenizer
    self.model_type = model_type
    self.stop_tokens = ["\ndef", "\nclass", "\n#"]
    if self.model_type == "huggingface":
      local_model_path = "/scratch/avani/qwen"  # from your snapshot_download

      self.tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
      self.model = AutoModelForCausalLM.from_pretrained(
      local_model_path,
      device_map="auto",             # automatically selects GPUs if available
      torch_dtype=torch.float16,     # for large models like 32B
      trust_remote_code=True
  )

  def _draw_sample(self, prompt: str) -> str:
    """Returns a predicted continuation of `prompt`."""
    if self.model_type == 'huggingface':
      try:
        llm = vLLM(model="/scratch/avani/gpt")
        params = SamplingParams(temperature=0.7, max_tokens=15000)
        prompt_addon = '''Your task:\n"
                "Return a **correct implementation** of the `make_stick` function in Python.\n\n"
                "Formatting Requirements (do NOT ignore):\n"
                "1. Your response MUST begin exactly like this:\n"
                "   ```python\n"
                "   def make_stick(env):\n"
                "2. Only output the complete function implementation inside the code block.\n\n"
                "Example of correct response format:\n"
                "```python\n"
                "def make_stick(env):\n"
                "    # your implementation here\n"
                "```\n"
                "Now return only the correct implementation of `make_stick` following these rules.'''
        prompt = prompt_addon + prompt
        
        output = llm.generate(prompt, params)
        response = output[0].outputs[0].text
        response = response[response.index("```python")+len("```python"):]
        response = response[:response.index("```")]
        return response
      #   inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
      #   while True:
          
      #     outputs = self.model.generate(
      #         **inputs,
      #         max_new_tokens=512,
      #         do_sample=True,
      #         temperature=0.8,
      #         top_p=0.9,
      #         pad_token_id=self.tokenizer.eos_token_id,
      #         eos_token_id=self.tokenizer.eos_token_id,
      #     )
      #     generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
          
      #     continuation = generated_text[len(prompt):]
          
      #     for stop_token in self.stop_tokens:
      #       if stop_token in continuation:
      #         continuation = continuation.split(stop_token)[0]
      #     print("continuation", continuation)
      #     if not is_reward_hacking_from_body(continuation):
      #       break
      #   return continuation
      except Exception as e:
        print(f"Error in Hugging Face generation: {e}")
        return "return [0]"
    elif self.model_type == 'gemini':
            prompt = "You must act as a code completion model that is completing the last function. Please only return code that will fit in that function. Do not imports or add the function signature on the top. Return only the code that will be inside the function." + prompt
            response = client.models.generate_content(
                  model="gemini-2.5-flash", contents = prompt
              )

            b = response.text
            if "```python" in b:
              b = extract_code_block(response.text)
            print("post extract", b)

            return b 

    else:  # ollama
      print("going in the llm")
      # prompt = "Only return the code completion of the function and nothing outside of this fucntion body followed by '''python tag.\n" + prompt 
      api_url = "http://129.128.243.184:11434/api/generate"
      headers = {"Content-Type": "application/json"}
      # print(prompt)
      try:
            #qwen2.5-coder:32b
            #gpt-oss:latest
            instruction = (
            "Act as a code completion model. "
            "Return only the function body of the most recent function definition (after its `def func_vn():` line). "
            "Do not include the `def func_vn():` line itself. "
            "Do not generate or suggest earlier versions (e.g., `collect_v0`, `collect_v1`, etc.). "
            "Only provide the body of the latest function exactly as written. "
            "Wrap the output inside `$$`. "
            "Example: `def func_v3():$ return None$` → output should be `$$return None$$`."
            )
            prompt = f"{prompt}/n/n{instruction}"
          
            payload = {
              "model": "gpt-oss:latest", 
              "prompt": prompt, 
              "template": "{{.Prompt}}",
              "stream": False, 
              "options": {
                "num_ctx": 4096, 
                # "stop": self.stop_tokens
              }
            }
            response = requests.post(api_url, headers=headers, json=payload, timeout=120)
            # print(response)
            b = extract_between_dollars(response.json()["response"])
            b = textwrap.dedent(b).strip()
            print(b)
            return b
      except Exception as e:
        print(f"Error in Ollama generation: {e}")
        return "return [0]"

  def draw_samples(self, prompt: str) -> Collection[str]:
    """Returns multiple predicted continuations of `prompt`."""
    return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]


class Sampler:
  """Node that samples program continuations and sends them for analysis."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      evaluators: Sequence[evaluator.Evaluator],
      samples_per_prompt: int,
      tokenizer=None,
      model_type="ollama"
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self.model_type = model_type
    self._llm = LLM(samples_per_prompt, tokenizer=tokenizer, model_type=model_type)

  def _get_function_signature(self, function_name: str) -> str:
    """Reads the function signature from the corresponding txt file."""
    try:
      with open(f"{function_name}.txt", 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
          if line.strip().startswith("@funsearch.evolve"):
            if i + 1 < len(lines):
              return lines[i + 1].strip()
      return f"def {function_name}(env) -> int:"  # Default fallback
    except Exception as e:
      print(f"Error reading function signature: {e}")
      return f"def {function_name}(env) -> int:"  # Default fallback

  def sample(self):
    """Continuously gets prompts, samples programs, sends them for analysis."""
    n=0
    best_samples=[]
    f = 0
    if self.model_type == "gemini":
      client = genai.Client()
    while n<500:
      prompt = self._database.get_prompt()
      # print(prompt)
      samples = self._llm.draw_samples(prompt.code)
      n+=1
      for sample in samples:
        chosen_evaluator = np.random.choice(self._evaluators)
        best = chosen_evaluator.analyse(
            sample, prompt.island_id, prompt.version_generated)
      #   if(best == 1):
      #     best_samples.append(sample)
      #   if(len(best_samples) >= 5):
      #     f = 1
      #     break
      # if(f == 1):
      #   break
    
    # Log best samples to file
    if best_samples:
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        function_name = self._evaluators[0]._function_name
        log_file = os.path.join(results_dir, f'best_samples_{current_date}_{function_name}.log')
        with open(log_file, 'w') as f:
            for i, sample in enumerate(best_samples, 1):
                f.write(f"Best Sample {i}:\n")
                f.write(self._get_function_signature(function_name) + "\n")
                # f.write('    """Generated function for program synthesis."""\n')
                f.write(sample)
                f.write("\n\n")


        
