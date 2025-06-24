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
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests
import os
from datetime import datetime

from funsearch.implementation import evaluator
from funsearch.implementation import programs_database


class LLM:
  """Language model that predicts continuation of provided source code."""

  def __init__(self, samples_per_prompt: int, model=None, tokenizer=None, model_type='ollama') -> None:
    self._samples_per_prompt = samples_per_prompt
    self.model = model
    self.tokenizer = tokenizer
    self.model_type = model_type
    self.stop_tokens = ["\ndef", "\nclass", "\n#", "\nimport"]

  def _draw_sample(self, prompt: str) -> str:
    """Returns a predicted continuation of `prompt`."""
    if self.model_type == 'huggingface':
      try:
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate continuation
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        continuation = generated_text[len(prompt):]
        
        for stop_token in self.stop_tokens:
          if stop_token in continuation:
            continuation = continuation.split(stop_token)[0]
        
        return continuation
      except Exception as e:
        print(f"Error in Hugging Face generation: {e}")
        return "return [0]"
    else:  # ollama
      print("going in the llm")
      api_url = "http://129.128.243.184:11434/api/generate"
      headers = {"Content-Type": "application/json"}
      try:
        payload = {
          "model": "qwen2.5-coder:32b", 
          "prompt": prompt, 
          "stream": False, 
          "template": "{{ .Prompt }}", 
          "options": {
            "num_ctx": 4096, 
            "stop": self.stop_tokens
          }
        }
        res = requests.post(api_url, headers=headers, json=payload, timeout=300)
        with open("prompt_eval.txt", 'w') as f:
            f.write(res.json()["response"])
        return res.json()["response"]
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
      model=None,
      tokenizer=None,
      model_type="ollama"
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self._llm = LLM(samples_per_prompt, model=model, tokenizer=tokenizer, model_type=model_type)

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
    while n<2000:
      prompt = self._database.get_prompt()
      # print(prompt)
      samples = self._llm.draw_samples(prompt.code)
      n+=1
      for sample in samples:
        chosen_evaluator = np.random.choice(self._evaluators)
        best = chosen_evaluator.analyse(
            sample, prompt.island_id, prompt.version_generated)
        if(best == 1):
          best_samples.append(sample)
        if(len(best_samples) >= 5):
          f = 1
          break
      if(f == 1):
        break
    
    # Log best samples to file
    if best_samples:
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        function_name = self._evaluators[0].function_name
        log_file = os.path.join(results_dir, f'best_samples_{current_date}_{function_name}.log')
        with open(log_file, 'w') as f:
            for i, sample in enumerate(best_samples, 1):
                f.write(f"Best Sample {i}:\n")
                f.write(self._get_function_signature(function_name) + "\n")
                # f.write('    """Generated function for program synthesis."""\n')
                f.write(sample)
                f.write("\n\n")


        
