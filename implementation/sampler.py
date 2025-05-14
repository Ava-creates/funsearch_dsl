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
import requests
import numpy as np

from funsearch.implementation import evaluator
from funsearch.implementation import programs_database


class LLM:
  """Language model that predicts continuation of provided source code."""

  def __init__(self, samples_per_prompt: int) -> None:
    self._samples_per_prompt = samples_per_prompt
    self._models = [
      "deepseek-coder-v2:16b",
      "codellama:13b",
      "mistral:7b",
      "llama2:13b"
    ]

  def _draw_sample(self, prompt: str) -> str:
    """Returns a predicted continuation of `prompt`."""
    print("going in the lllm")
    api_url = "http://129.128.243.184:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    try:
      payload = {"model": "qwen2.5-coder:32b", "prompt": prompt, "stream": False, "template": "{{ .Prompt }}", "options": {"num_ctx": 4096, "stop": ["\ndef", "\nclass", "\n#", "\nimport"]}}
      res = requests.post(api_url, headers=headers, json=payload, timeout=300)
    # print(res.json()["response"])
      w=  res.json()["response"]
      # print(w)
      return w
    except:
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
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self._llm = LLM(samples_per_prompt)

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
