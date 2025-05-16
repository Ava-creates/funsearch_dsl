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

from funsearch.implementation import evaluator
from funsearch.implementation import programs_database


class LLM:
  """Language model that predicts continuation of provided source code."""

  def __init__(self, samples_per_prompt: int, model=None, tokenizer=None) -> None:
    self._samples_per_prompt = samples_per_prompt
    self.model = model
    self.tokenizer = tokenizer
    self.stop_tokens = ["\ndef", "\nclass", "\n#", "\nimport"]

  def _draw_sample(self, prompt: str) -> str:
    """Returns a predicted continuation of `prompt`."""
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
      print(f"Error during generation: {str(e)}")
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
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self._llm = LLM(samples_per_prompt, model=model, tokenizer=tokenizer)

  def sample(self):
    """Continuously gets prompts, samples programs, sends them for analysis."""
    n=0
    while n<2000:
      prompt = self._database.get_prompt()
      samples = self._llm.draw_samples(prompt.code)
      n+=1
      for sample in samples:
        # print("sample from the llm", sample)
        chosen_evaluator = np.random.choice(self._evaluators)
        chosen_evaluator.analyse(
            sample, prompt.island_id, prompt.version_generated)
