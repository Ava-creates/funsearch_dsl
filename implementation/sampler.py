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
import os
import sys

def _import_vllm():
    from vllm import LLM as vLLM
    from vllm import SamplingParams
    return vLLM, SamplingParams


def _import_auto_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.utils.openai_compat_key import resolve_openai_compat_api_key
from src.utils.api_openai_compat_walltimes import env_float_openai_compat_scaled
# from google import genai
from funsearch.implementation import evaluator
from funsearch.implementation import programs_database

import ast
import textwrap
import re
import json

def extract_between_dollars(text: str) -> str:
    """
    Extracts the substring between the first pair of $$ ... $$.
    Returns an empty string if not found.
    """
    start = text.find("$$")
    if start == -1:
        # Preserve leading indentation for parser compatibility; trim only line breaks.
        return text.strip("\r\n")
    end = text.find("$$", start + 2)
    if end == -1:
        return text.strip("\r\n")

    # IMPORTANT: Do not call .strip() here. It removes leading spaces from the
    # first code line (e.g., "  x = 1" -> "x = 1"), which causes
    # "expected an indented block" when evaluator wraps this as a function body.
    return text[start + 2:end].strip("\r\n")

def extract_code_block(text):
    """Extracts the first Python code block enclosed in triple backticks."""
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    return match.group(1) if match else None


def _normalize_unicode_quotes(text: str) -> str:
    """Normalize Unicode quotes and other problematic characters to ASCII equivalents."""
    replacements = {
        '\u2018': "'",  # Left single quotation mark
        '\u2019': "'",  # Right single quotation mark
        '\u201A': "'",  # Single low-9 quotation mark
        '\u201B': "'",  # Single high-reversed-9 quotation mark
        '\u201C': '"',  # Left double quotation mark
        '\u201D': '"',  # Right double quotation mark
        '\u201E': '"',  # Double low-9 quotation mark
        '\u201F': '"',  # Double high-reversed-9 quotation mark
        '\u2032': "'",  # Prime
        '\u2033': '"',  # Double prime
    }
    result = text
    for unicode_char, ascii_char in replacements.items():
        result = result.replace(unicode_char, ascii_char)
    return result


def is_reward_hacking_from_body(code_body: str) -> bool:
    # Normalize Unicode quotes before parsing
    code_body = _normalize_unicode_quotes(code_body)
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

  def __init__(self, samples_per_prompt: int, model=None, tokenizer=None, model_type='vllm', function_name=None, shared_vllm=None, vllm_lock=None) -> None:
    self._samples_per_prompt = samples_per_prompt
    self.tokenizer = tokenizer
    # Backward compatibility: existing callers may still pass "huggingface".
    self.model_type = "vllm" if model_type == "huggingface" else model_type
    if self.model_type not in {"vllm", "openai_compat"}:
      raise ValueError(
          f"Unsupported model_type={model_type!r}. Allowed: 'vllm' (or alias 'huggingface') and 'openai_compat'."
      )
    self.stop_tokens = ["\ndef", "\nclass", "\n#"]
    self.vllm_lock = vllm_lock  # Thread lock for safe concurrent access to shared vLLM
    self.llm = None
    self.params = None
    self._hf_tokenizer = None
    if self.model_type == "openai_compat":
      return
    AutoTokenizer = _import_auto_tokenizer()
    self._hf_tokenizer = AutoTokenizer.from_pretrained(
        "/scratch/avani/gpt",
        trust_remote_code=True,
        local_files_only=True,
    )
    print("Using HuggingFace tokenizer for prompt token counting (/scratch/avani/gpt)")
    if self.model_type == "vllm":
      vLLM, SamplingParams = _import_vllm()
      # Use shared vLLM instance if provided, otherwise create new one
      if shared_vllm is not None:
        self.llm = shared_vllm
        print("Using shared vLLM instance in sampler")
      else:
        # Fallback: create new instance if shared_vllm not provided
        # Use lower memory utilization since multiple jobs may run in parallel
        self.llm = vLLM(
            model="/scratch/avani/gpt", 
            tensor_parallel_size=4,
            gpu_memory_utilization=0.6  # Reduced to 60% to handle parallel jobs
        )
        print("Created new vLLM instance in sampler (shared_vllm not provided) - using reduced memory settings")
      self.params = SamplingParams(temperature=0.7, max_tokens=35000)
      # local_model_path = "/scratch/avani/qwen"  # from your snapshot_download

      # self.tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
      # self.model = AutoModelForCausalLM.from_pretrained(
      # local_model_path,
      # device_map="auto",             # automatically selects GPUs if available
      # torch_dtype=torch.float16,     # for large models like 32B
      # trust_remote_code=True

  def _build_prompt_for_model(self, prompt: str, function_name: str) -> str:
    """Builds the exact prompt string sent to the model."""
    if self.model_type == "vllm":
      prompt_addon = f"Your task:\nReturn the **body** of the `{function_name}_vn` function in Python.\n\nFormatting Requirements (do NOT ignore):\n1. Your response MUST begin exactly like this:\n   ```python\n2. Your response MUST end with a closing triple backtick (```).\n3. DO NOT include the function definition line (`def {function_name}_vn(env):`).\n4. DO NOT include any explanations, markdown text, or comments outside the code.\n5. Inside the code block, include only valid Python statements that belong inside the function body.\n6. The code must be properly indented for direct insertion after:\n       def {function_name}_vn(env):\n7. Output must contain **only** the code block — no text before or after it."
      return prompt_addon + prompt
    instruction = (
        "Act as a code completion model. "
        "Return only the function body of the most recent function definition (after its `def {function_name}_vn():` line). "
        "Do not include the `def {function_name}_vn():` line itself. "
        "Only provide the body of the latest function exactly as written. "
        "Wrap the output inside `$$`. "
        "Example: `def {function_name}_vn():$ return None$` → output should be `$$return None$$`."
    )
    return f"{prompt}/n/n{instruction}"

  def count_prompt_tokens(self, prompt: str, function_name: str) -> int:
    """Returns /scratch/avani/gpt tokenizer token count for the exact model prompt."""
    full_prompt = self._build_prompt_for_model(prompt, function_name)
    if self._hf_tokenizer is None:
      return max(1, len(full_prompt) // 4)
    return len(self._hf_tokenizer.encode(full_prompt, add_special_tokens=False))

  def _draw_sample(self, prompt: str, function_name: str) -> str:
    """Returns a predicted continuation of `prompt`."""
    if self.model_type == 'vllm':
      try:
        prompt = self._build_prompt_for_model(prompt, function_name)
        # print(prompt)
        # Use lock for thread-safe access to shared vLLM instance
        if self.vllm_lock is not None:
          with self.vllm_lock:
            output = self.llm.generate(prompt, self.params)
        else:
          output = self.llm.generate(prompt, self.params)
        response = output[0].outputs[0].text
        response = response[response.index("```python")+len("```python"):]
        response = response[:response.index("```")]
        # print(response)
        # response = response[response.index(":")+1:]
       # response = textwrap.dedent(response).strip()
        return  response
  
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
    if self.model_type == "openai_compat":
      from src.utils.openai_compat_http import extract_chat_content, post_chat_completion

      prompt = self._build_prompt_for_model(prompt, function_name)
      model_name = os.environ.get("OPENAI_COMPAT_MODEL", "gpt-oss-120b").strip()
      timeout_seconds = env_float_openai_compat_scaled("OPENAI_COMPAT_HTTP_TIMEOUT", 300.0)
      print(
        f"[OpenAICompat][{function_name}] request model={model_name} timeout={timeout_seconds}s"
      )
      status, raw_text, body = post_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=10000,
        timeout_seconds=timeout_seconds,
        label=f"OpenAICompat[{function_name}]",
      )
      if status >= 400 or body is None:
        print(f"OpenAI-compatible API error {status}: {raw_text[:500]}")
        return "return [0]"
      content = extract_chat_content(body)
      if not content:
        print("OpenAI-compatible API returned no choices")
        return "return [0]"
      print(extract_between_dollars(content))
      return extract_between_dollars(content)
    raise ValueError(f"Unexpected model_type={self.model_type!r}")

  def draw_samples(self, prompt: str, function_name: str) -> Collection[str]:
    """Returns multiple predicted continuations of `prompt`."""
    return [self._draw_sample(prompt, function_name) for _ in range(self._samples_per_prompt)]


class Sampler:
  """Node that samples program continuations and sends them for analysis."""

  # Run-level cache shared by all Sampler instances in the same process.
  _cached_early_stop_target: float | None = None
  _cached_early_stop_key: tuple[str, ...] | None = None

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      evaluators: Sequence[evaluator.Evaluator],
      samples_per_prompt: int,
      tokenizer=None,
      model_type="vllm",
      shared_vllm=None,
      vllm_lock=None,
      num_iterations: int = 500
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self.model_type = model_type
    self._llm = LLM(samples_per_prompt, tokenizer=tokenizer, model_type=model_type, shared_vllm=shared_vllm, vllm_lock=vllm_lock)
    self._function_name = os.path.splitext(os.path.basename(self._evaluators[0]._function_name))[0]
    # Use round-robin assignment for even distribution of work across evaluators
    self._evaluator_index = 0
    self._num_iterations = num_iterations
    self._logged_first_prompt_tokens = False
    self._early_stop_target = self._get_or_compute_early_stop_target()
    print(f"[Sampler] Early-stop target reward: {self._early_stop_target}")

  def _current_early_stop_key(self) -> tuple[str, ...]:
    """Stable key for the testcase set used in this sampler run."""
    if not self._evaluators:
      return tuple()
    template_source = str(self._evaluators[0]._template)
    grid_paths = self._extract_grid_paths_from_template(template_source)
    return tuple(sorted(str(p) for p in grid_paths))

  def _get_or_compute_early_stop_target(self) -> float:
    """Compute once per run and reuse across samplers."""
    current_key = self._current_early_stop_key()
    if (
        Sampler._cached_early_stop_target is not None
        and Sampler._cached_early_stop_key == current_key
    ):
      return Sampler._cached_early_stop_target

    target = self._compute_early_stop_target()
    Sampler._cached_early_stop_target = target
    Sampler._cached_early_stop_key = current_key
    return target

  def _extract_grid_paths_from_template(self, template_source: str) -> list[str]:
    """Extract embedded _grid_spec_paths list from evaluate() source, if present."""
    match = re.search(r"_grid_spec_paths\s*=\s*(\[[\s\S]*?\])", template_source)
    if not match:
      return []
    try:
      parsed = ast.literal_eval(match.group(1))
      if isinstance(parsed, list):
        return [str(p) for p in parsed]
    except Exception:
      return []
    return []

  def _compute_early_stop_target(self) -> float:
    """Compute testcase max reward: 100*positive + 1*other.

    Falls back to historical default (1025) when testcase metadata is unavailable.
    """
    if not self._evaluators:
      return 1025.0

    try:
      template_source = str(self._evaluators[0]._template)
      grid_paths = self._extract_grid_paths_from_template(template_source)
      if not grid_paths:
        return 1025.0

      positive_count = 0
      other_count = 0
      for path in grid_paths:
        try:
          with open(path, "r", encoding="utf-8") as f:
            spec = json.load(f)
          test_type = str(spec.get("test_type", "positive")).lower()
          if test_type == "positive":
            positive_count += 1
          else:
            other_count += 1
        except Exception:
          # Treat unreadable/malformed testcase files as non-positive for conservative thresholding.
          other_count += 1

      return float(100 * positive_count + 1 * other_count)
    except Exception:
      return 1025.0


  def _get_function_signature(self, function_name: str) -> str:
    """Reads the function signature from the corresponding txt file."""
    self._function_name = function_name
    try:
      with open(f"{function_name}.txt", 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
          if line.strip().startswith("@funsearch.evolve"):
            if i + 1 < len(lines):
              return lines[i + 1].strip()
        self._function_name = function_name
      return f"def {function_name}(env) -> int:"  # Default fallback
    except Exception as e:
      print(f"Error reading function signature: {e}")
      return f"def {function_name}(env) -> int:"  # Default fallback

  def sample(self):
    """Continuously gets prompts, samples programs, sends them for analysis."""
    n=0
    best_samples=[]
    f = 0
    best = 0 
    while n < self._num_iterations:
      prompt = self._database.get_prompt()
      samples = self._llm.draw_samples(prompt.code, self._function_name)
      n+=1
      for sample in samples:
        if not self._logged_first_prompt_tokens:
          prompt_tokens = self._llm.count_prompt_tokens(prompt.code, self._function_name)
          print(f"[Sampler] First prompt token count: {prompt_tokens}")
          self._logged_first_prompt_tokens = True

        chosen_evaluator = self._evaluators[self._evaluator_index % len(self._evaluators)]
        self._evaluator_index += 1
        best = max(best, chosen_evaluator.analyse(
            sample, prompt.island_id, prompt.version_generated))
      #early stopping
      if best >= self._early_stop_target:
        print("early stopping with target", self._early_stop_target)
        break


        
