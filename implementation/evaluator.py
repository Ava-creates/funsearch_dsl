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

"""Class for evaluating programs proposed by the Sampler."""
import ast
import textwrap
from collections.abc import Sequence
import copy
from typing import Any
import tempfile
import os
import subprocess
from typing import Any, Tuple
import json
from datetime import datetime
import time
import threading

# Module-level lock for thread-safe log file writes
_log_file_lock = threading.Lock()

from funsearch.implementation import code_manipulation
from funsearch.implementation import programs_database
import re


class _FunctionLineVisitor(ast.NodeVisitor):
  """Visitor that finds the last line number of a function with a given name."""

  def __init__(self, target_function_name: str) -> None:
    self._target_function_name: str = target_function_name
    self._function_end_line: int | None = None

  def visit_FunctionDef(self, node: Any) -> None:  # pylint: disable=invalid-name
    """Collects the end line number of the target function."""
    if node.name == self._target_function_name:
      self._function_end_line = node.end_lineno
    self.generic_visit(node)

  @property
  def function_end_line(self) -> int:
    """Line number of the final line of function `target_function_name`."""
    assert self._function_end_line is not None  # Check internal correctness.
    return self._function_end_line


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


def _trim_function_body(generated_code: str) -> tuple[str, bool]:
  """Extracts the body of the generated function, trimming anything after it.
  
  Returns:
    (body, had_syntax_error)
  """
  if not generated_code:
    return '', False
  # Normalize Unicode quotes before parsing
  generated_code = _normalize_unicode_quotes(generated_code)
  code = f'def fake_function_header():\n{generated_code}'
  tree = None
  had_syntax_error = False
  # We keep trying and deleting code from the end until the parser succeeds.
  while tree is None:
    try:
      tree = ast.parse(code)
    except SyntaxError as e:
      print(e)
      had_syntax_error = True
      code = '\n'.join(code.splitlines()[:e.lineno - 1])
  if not code:
    # Nothing could be saved from `generated_code`
    return '', had_syntax_error

  visitor = _FunctionLineVisitor('fake_function_header')
  visitor.visit(tree)
  body_lines = code.splitlines()[1:visitor.function_end_line]
  return '\n'.join(body_lines) + '\n\n', had_syntax_error


def _sample_to_program(
    generated_code: str,
    version_generated: int | None,
    template: code_manipulation.Program,
    function_to_evolve: str,
) -> tuple[code_manipulation.Function, str, bool]:
  """Returns the compiled generated function and the full runnable program."""
  # print(generated_code)
  body, had_syntax_error = _trim_function_body(generated_code)
  # print(body)
  if version_generated is not None:
    body = code_manipulation.rename_function_calls(
        body,
        f'{function_to_evolve}_v{version_generated}',
        function_to_evolve)

  program = copy.deepcopy(template)
  evolved_function = program.get_function(function_to_evolve)
  evolved_function.body = textwrap.indent(textwrap.dedent(body).strip(), "  ")
 # evolved_function.body = body
  return evolved_function, str(program), had_syntax_error


class Sandbox:
  """Sandbox for executing generated code."""

  def run(
        self,
        program: str,
        function_to_run: str,
        test_input: Any,
        timeout_seconds: int,
        function_name: str = None
    ) -> Tuple[Any, bool]:
            """
            Executes Python code in a subprocess and returns:
            - The function's output
            - Boolean indicating successful execution
            """
            # Create unique filename using function name, process ID and timestamp
            # Include function name to prevent collisions when parallel funsearch workers run
            temp_dir = os.getcwd()
            unique_id = f"{os.getpid()}_{int(time.time() * 1000000)}"
            if function_name:
                # Sanitize function name for use in filename (drop extensions like .txt/.py)
                base_name = os.path.splitext(os.path.basename(function_name))[0]
                safe_func_name = base_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
                script_path = f'generated_code_{safe_func_name}_{unique_id}.py'
            else:
                script_path = f'generated_code_{unique_id}.py'
            script_path = os.path.join(temp_dir, script_path)

            
            # Create complete executable program
            # Store result and print in a way that doesn't print grid tables
            # Build the program by concatenating strings to avoid f-string issues with triple quotes
            json_code = f"""
import json
import sys
result = {function_to_run}()
# Convert result to JSON-serializable format, but keep grid strings
# Use json.dumps to avoid printing markdown tables
if isinstance(result, list):
    # Convert numpy types to native Python types for JSON serialization
    import numpy as np
    total_reward = result[0] if len(result) > 0 else 0.0
    if isinstance(total_reward, (np.integer, np.floating)):
        total_reward = float(total_reward)
    else:
        total_reward = float(total_reward) if total_reward is not None else 0.0
    
    actions_count = result[1] if len(result) > 1 else 0
    if isinstance(actions_count, (np.integer, np.floating)):
        actions_count = int(actions_count)
    else:
        actions_count = int(actions_count) if actions_count is not None else 0
    
    # Domain templates now always return:
    # [total_reward, actions_count, ans, grid_after]
    ans = result[2] if len(result) > 2 else None
    if isinstance(ans, np.ndarray):
        ans = ans.tolist()
    elif isinstance(ans, tuple):
        ans = list(ans)
    grid_before = None

    result_dict = {{
        'total_reward': total_reward,
        'actions_count': actions_count,
        'ans': ans,
        'grid_before': grid_before,
        'grid_after': result[3] if len(result) > 3 and result[3] else None
    }}
    # Print as JSON to avoid markdown table printing
    print(json.dumps(result_dict))
else:
    print(json.dumps({{'total_reward': 0, 'actions_count': 0, 'ans': None, 'grid_before': None, 'grid_after': None}}))
"""
            env_vars = os.environ.copy()
            env_vars["PYTHONPATH"] = (
                "/project/aip-lelis/avani/DSL_Generator"
                + os.pathsep
                + env_vars.get("PYTHONPATH", "")
            )

            # Concatenate program and json_code to avoid f-string issues with triple quotes in program
            full_program = program + json_code
            # Normalize Unicode quotes before writing to file (Python will parse this file)
            full_program = _normalize_unicode_quotes(full_program)
            # print(full_program)
            try:
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(full_program)

                # Convert input to string representation
                input_str = str(test_input)
                
                # Execute in subprocess with timeout
                result = subprocess.run(
                    ['python3', script_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=True,
                    encoding='utf-8',
                    errors='replace',
                    env=env_vars
                )

                output = result.stdout.strip()
                
                # DEBUG: Print raw output from subprocess
                print(f"[DEBUG Sandbox] Raw stdout from subprocess: {output[:500]}")  # First 500 chars
                if result.stderr:
                    print(f"[DEBUG Sandbox] stderr: {result.stderr[:500]}")
                
                # Parse JSON output
                import json
                try:
                    result_dict = json.loads(output)
                    total_reward = result_dict.get('total_reward', 0.0)
                    actions_count = result_dict.get('actions_count', 0)
                    ans = result_dict.get('ans')
                    grid_before = result_dict.get('grid_before')
                    grid_after = result_dict.get('grid_after')
                    print(f"[DEBUG Sandbox] Parsed JSON - total_reward: {total_reward}, actions_count: {actions_count}")
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"[DEBUG Sandbox] JSON parse failed: {e}, trying fallback parsing")
                    # Fallback to old parsing method
                    output = output.replace("np.float64", "")
                    output = output.replace("np.float32", "")
                    output = output.replace("(", "").replace(")", "")
                    parsed = ast.literal_eval(output)
                    if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                        total_reward = parsed[0]
                        actions_count = parsed[1]
                        ans = parsed[2] if len(parsed) > 2 else None
                        grid_before = None
                        grid_after = parsed[3] if len(parsed) > 3 else None
                        print(f"[DEBUG Sandbox] Fallback parsed - total_reward: {total_reward}, actions_count: {actions_count}")
                    else:
                        raise ValueError(f"Unexpected parsed format: {parsed}")

                # print("total_reward:", total_reward)
                # print("actions_count:", actions_count)

                try:
                    return_value = (float(total_reward), True, actions_count, grid_before, grid_after, ans)
                    print(f"[DEBUG Sandbox] Returning: {return_value}")
                    return return_value
                except ValueError:
                    print("Output is not a float.")
                    return -1, True, 0, None, None, None
                    
            except subprocess.TimeoutExpired:
                return -1, False, 0, None, None, None
            except subprocess.CalledProcessError as e:
                print(f"Process Error: Command failed with exit code {e.returncode}")
                print(f"Command: {e.cmd}")
                print(f"Output: {e.stdout}")
                print(f"Error: {e.stderr}")
                return -1, False, 0, None, None, None
            except Exception as e:
                print(f"Unexpected Error: {e}")
                return -1, False, 0, None, None, None 
            finally:
                # Clean up the temporary file (best-effort, ignore races)
                try:
                    if os.path.exists(script_path):
                        os.remove(script_path)
                except OSError:
                    pass
                # Best-effort cleanup for any stray generated_code_*.py files
                try:
                    temp_dir = os.getcwd()
                    for fname in os.listdir(temp_dir):
                        if fname.startswith("generated_code_") and fname.endswith(".py"):
                            try:
                                os.remove(os.path.join(temp_dir, fname))
                            except OSError:
                                pass
                except OSError:
                    pass


def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
  """Returns whether the generated function is calling an earlier version."""
  for name in code_manipulation.get_functions_called(program):
    # In `program` passed into this function the most recently generated
    # function has already been renamed to `function_to_evolve` (wihout the
    # suffix). Therefore any function call starting with `function_to_evolve_v`
    # is a call to an ancestor function.
    if name.startswith(f'{function_to_evolve}_v'):
      return True
  return False


def format_grid_state_from_markdown(grid_markdown: str) -> str:
  """Format grid markdown for LLM prompt."""
  if grid_markdown:
    return f"Grid State:\n{grid_markdown}\n"
  return "Grid State: Not available\n"


# Global vLLM instance for LLM verifier (lazy initialization, fallback if shared_vllm not provided)
_llm_verifier_instance = None
_llm_verifier_params = None

def _get_llm_verifier(shared_vllm=None):
  """Get or create the vLLM instance for verification.
  
  Args:
    shared_vllm: Optional shared vLLM instance from FunSearch
  """
  global _llm_verifier_instance, _llm_verifier_params
  # Use shared vLLM if provided, otherwise use global instance
  if shared_vllm is not None:
    if _llm_verifier_params is None:
      from vllm import SamplingParams
      _llm_verifier_params = SamplingParams(temperature=0.5, max_tokens=15000)
    return shared_vllm, _llm_verifier_params
  
  # Fallback to global instance if no shared_vllm provided
  if _llm_verifier_instance is None:
    try:
      from vllm import LLM as vLLM
      from vllm import SamplingParams
      _llm_verifier_instance = vLLM(
          model="/scratch/avani/gpt", 
          tensor_parallel_size=4,
          gpu_memory_utilization=0.6  # Reduced to 60% to handle parallel jobs
      )
      _llm_verifier_params = SamplingParams(temperature=0.3, max_tokens=100)
    except Exception as e:
      print(f"Failed to initialize vLLM for verifier: {e}")
      return None, None
  return _llm_verifier_instance, _llm_verifier_params

def call_llm_verifier(function_description: str, 
                     grid_before: str, grid_after: str, 
                     function_args: str = "", shared_vllm=None, vllm_lock=None) -> float:
  """Call an LLM to verify function behavior and return a reward.
  
  Args:
    function_description: Description of what the function should do
    grid_before: Grid state before function execution
    grid_after: Grid state after function execution
    function_args: Arguments passed to the function
    
  Returns:
    Float reward value (-1.0 to 1.0)
  """
  try:
    # Get vLLM instance (use shared_vllm if provided)
    llm, params = _get_llm_verifier(shared_vllm)
    if llm is None or params is None:
      return 0.0
    
    prompt = f"""You are a reward verifier for a program synthesis system. Evaluate whether a function correctly performed its intended behavior.

Function Description:
{function_description}

Function Arguments:
{function_args}

Grid State BEFORE function execution:
{grid_before}

Grid State AFTER function execution:
{grid_after}

Task: Analyze whether the function correctly performed its intended behavior based on the grid state changes. Consider:
1. Did the function produce the expected state changes?
2. Are the changes consistent with the function description?
3. Did the function handle the arguments correctly?

IMPORTANT: If the grid state is UNCHANGED (grid_before == grid_after), this is NOT acceptable. The function must produce some change to the grid state. An unchanged grid should receive a negative reward (typically -0.5 or lower) unless the function description explicitly states it should not modify the state.

Return ONLY a single float value between -1.0 and 1.0 representing the reward:
- 1.0: Perfect execution, exactly as described
- 0.5 to 0.9: Good execution with minor issues
- 0.0 to 0.4: Partial success
- Negative values: Incorrect or harmful behavior, including unchanged grid when changes are expected

Return format: Just the float number, nothing else.
"""
    
    # Call vLLM with thread-safe locking if lock is provided
    if vllm_lock is not None:
      with vllm_lock:
        output = llm.generate([prompt], sampling_params=params)
    else:
      output = llm.generate([prompt], sampling_params=params)
    result_text = output[0].outputs[0].text.strip()
    # print("result_text pre parsing:", result_text)
    # GPT-OSS may return reasoning before the final answer
    # Look for "assistantfinal" marker and parse after it
    assistant_final_marker = "assistantfinal"
    if assistant_final_marker.lower() in result_text.lower():
      # Find the position after "assistantfinal" (case-insensitive)
      marker_pos = result_text.lower().find(assistant_final_marker.lower())
      if marker_pos != -1:
        # Extract text after the marker
        after_marker = result_text[marker_pos + len(assistant_final_marker):].strip()
        # Remove any colons or whitespace separators that might follow (but preserve negative signs)
        after_marker = re.sub(r'^[:\s]+', '', after_marker)
        result_text = after_marker
    
    # Extract float from response
    try:
      # Try to parse the entire result_text as float first
      reward = float(result_text)
      # print("reward:", reward)
      # Clamp to reasonable range
      return max(-1.0, min(1.0, reward))
    except ValueError:
      # print(result_text)
      # print("reward is not a float")
      return 0.0
      
  except Exception as e:
    # If LLM call fails, return neutral reward
    print(f"LLM verifier error: {e}")
    return 0.0


def extract_function_description(specification: str, function_name: str) -> str:
  """Extract function description from specification."""
  try:
    # Look for @funsearch.evolve decorator followed by function definition
    pattern = rf'@funsearch\.evolve\s+def\s+{re.escape(function_name)}\s*\([^)]*\)[^:]*:\s*"""(.*?)"""'
    match = re.search(pattern, specification, re.DOTALL)
    if match:
      return match.group(1).strip()
    
    # Fallback: look for function docstring
    pattern2 = rf'def\s+{re.escape(function_name)}\s*\([^)]*\)[^:]*:\s*"""(.*?)"""'
    match2 = re.search(pattern2, specification, re.DOTALL)
    if match2:
      return match2.group(1).strip()
    
    return f"Function {function_name} implementation"
  except Exception:
    return f"Function {function_name} implementation"


class Evaluator:
  """Class that analyses functions generated by LLMs."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      template: code_manipulation.Program,
      model_name: str,
      function_to_evolve: str,
      function_to_run: str,
      inputs,
      function_to_init, 
      specification,
      function_name,
      experiment_dir: str = None,
      shared_vllm = None,
      vllm_lock = None,
      results_tracker = None,
      log_file: str = None  # Optional shared log file path for all evaluators
  ):
    self._database = database
    self._template = template
    self._function_to_evolve = function_to_evolve
    self._function_to_run = function_to_run
    self._inputs = inputs
    self._timeout_seconds = 300 #100 sec for each env difficulty
    self._function_name = function_name
    self._sandbox = Sandbox()
    self._log_file = log_file  # Use shared log file if provided, otherwise initialize on first use
    self._model_name = model_name
    self._function_init = function_to_init
    self.specification = specification
    self._experiment_dir = experiment_dir
    self._shared_vllm = shared_vllm  # Store shared vLLM instance
    self._vllm_lock = vllm_lock  # Thread lock for safe concurrent access to shared vLLM
    self._results_tracker = results_tracker  # Results tracker for interaction tracking

  def _build_log_file_path(self) -> str:
    """Build shared funsearch log path."""
    if self._experiment_dir:
      results_dir = self._experiment_dir
    else:
      results_dir = 'results/funsearch'
    os.makedirs(results_dir, exist_ok=True)
    current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    safe_function_name = os.path.basename(self._function_name).replace("/", ":").replace("\\", ":")
    safe_function_init = os.path.basename(self._function_init).replace("/", ":").replace("\\", ":")
    safe_spec = os.path.basename(self.specification).replace("/", "").replace("\\", "")
    return os.path.join(
        results_dir,
        f'{self._model_name}_q2.5_{safe_function_name}_{safe_function_init}_{safe_spec}_{current_date}.log'
    )

  def _ensure_log_paths(self) -> None:
    """Initialize primary log path once."""
    if self._log_file is None:
      self._log_file = self._build_log_file_path()

  def _append_json_line(self, path: str, payload: dict) -> None:
    """Thread-safe JSONL append helper."""
    with _log_file_lock:
      with open(path, 'a') as f:
        f.write(json.dumps(payload) + '\n')

  def analyse(
      self,
      sample: str,
      island_id: int | None,
      version_generated: int | None,
  ) -> None:
    """Compiles the sample into a program and executes it on test inputs."""
    # print("sample from the llm", sample)
    new_function, program, had_syntax_error = _sample_to_program(
        sample, version_generated, self._template, self._function_to_evolve)
    self._ensure_log_paths()
    # print("program ", program)

    # Check if function body is empty or only whitespace
    # new_function is a Function object, so check its body attribute
    if had_syntax_error:
      # Surface syntax errors as a hard failure with score -1
      scores_per_test = {}
      for current_input in self._inputs:
        scores_per_test[current_input] = -1
      log_entry = {
          'timestamp': datetime.now().isoformat(),
          'status': 'syntax_error',
          'invalid_reason': 'syntax_error',
          'version_generated': version_generated,
          'raw_sample': sample,
          'function_name': new_function.name if new_function else None,
          'function_body': new_function.body if new_function else None,
          "env_interactions": 0,
          'island_id': island_id,
          'scores': scores_per_test,
          'syntax_error': True
      }
      self._append_json_line(self._log_file, log_entry)
      self._database.register_program(new_function, island_id, scores_per_test)
      return 0
    if not new_function or not hasattr(new_function, 'body') or not new_function.body or not new_function.body.strip():
      scores_per_test = {}
      for current_input in self._inputs:
        scores_per_test[current_input] = -1
      log_entry = {
          'timestamp': datetime.now().isoformat(),
          'status': 'empty_body',
          'invalid_reason': 'empty_body',
          'version_generated': version_generated,
          'raw_sample': sample,
          'function_name': new_function.name if new_function else None,
          'function_body': new_function.body if (new_function and hasattr(new_function, 'body')) else sample,
          "env_interactions": 0,
          'island_id': island_id,
          'scores': scores_per_test,
          'empty_body': True
      }
      self._append_json_line(self._log_file, log_entry)
      return 0

    # Check if the function actually uses its non-env arguments.
    # If none of the domain-specific parameters (e.g. item, workshop, direction,
    # primitive) appear anywhere in the body, the function is ignoring its
    # arguments and its score is forced to 0.
    _args_ignored = False
    if new_function and hasattr(new_function, 'args') and new_function.args and new_function.body:
      _param_names = [
          p.strip() for p in new_function.args.split(',')
          if p.strip() and p.strip() != 'env'
      ]
      if _param_names:
        # Use AST to find actual variable references (ast.Name nodes).
        # This ignores comments, string literals, and substring matches
        # like "workshop" inside "workshop_indices".
        _used_names = set()
        _wrapper = f"def _f({new_function.args}):\n"
        _wrapper += textwrap.indent(new_function.body, "  ")
        print("wrapper", _wrapper)
        _tree = ast.parse(_wrapper)
        for _node in ast.walk(_tree):
          if isinstance(_node, ast.Name):
            _used_names.add(_node.id)
        _any_used = any(pname in _used_names for pname in _param_names)
        if not _any_used:
          _args_ignored = True
          print(f"[Evaluator] Argument-usage check: function '{new_function.name}' "
                f"does not reference any of its parameters {_param_names} — score forced to 0")

    scores_per_test = {}
    invalid_reasons = {}
    runs_ok = False
    env_interactions = 0
    # Extract function description from specification for LLM verifier
    function_description = extract_function_description(self.specification, self._function_to_evolve)
    
    # print("function to run", self._function_to_run)
    ancestor_called = _calls_ancestor(program, self._function_to_evolve)
    for current_input in self._inputs:
      result = self._sandbox.run(
          program, self._function_to_run, current_input, self._timeout_seconds,
          function_name=self._function_name)
      
      # Unpack result: (test_output, runs_ok, actions_count, grid_before, grid_after, ans)
      ans = None
      if len(result) >= 6:
        test_output, runs_ok, env_interactions, grid_before, grid_after, ans = result
      elif len(result) >= 5:
        test_output, runs_ok, env_interactions, grid_before, grid_after = result
      else:
        # No legacy formats supported; mark as failed run
        test_output, runs_ok, env_interactions, grid_before, grid_after, ans = -1, False, 0, None, None, None

      # DEBUG: Print result unpacking details
      print(f"[DEBUG] Result tuple length: {len(result)}")
      print(f"[DEBUG] Unpacked - test_output: {test_output}, runs_ok: {runs_ok}, env_interactions: {env_interactions}")
      print(f"[DEBUG] Result tracker available: {self._results_tracker is not None}")
      if self._results_tracker is not None:
        print(f"[DEBUG] Current funsearch interactions: {self._results_tracker.interactions.get('funsearch', 0)}")

      # Track funsearch interactions if tracker is available
      if self._results_tracker is not None and env_interactions > 0:
        print(f"[DEBUG] Tracking {env_interactions} funsearch interactions")
        self._results_tracker.add_funsearch_interactions(env_interactions)
        print(f"[DEBUG] After tracking - funsearch interactions: {self._results_tracker.interactions.get('funsearch', 0)}")
      elif self._results_tracker is not None:
        print(f"[DEBUG] NOT tracking - env_interactions is {env_interactions} (must be > 0)")
      else:
        print(f"[DEBUG] NOT tracking - results_tracker is None")
      z = 0
      invalid_reason = None
      if runs_ok == False:
        invalid_reason = "runtime_failure"
      elif ancestor_called:
        invalid_reason = "ancestor_call"
      elif test_output is None:
        invalid_reason = "missing_test_output"
      elif not isinstance(test_output, (int, float)):
        invalid_reason = "non_numeric_output"
      elif test_output <= -1:
        invalid_reason = "negative_test_output"

      if invalid_reason is not None:
        scores_per_test[current_input] = -1
        invalid_reasons[current_input] = invalid_reason
        z = -1
      elif _args_ignored:
        scores_per_test[current_input] = 0
        z = 0
      else:
        # Pure pass_check / environment reward; no LLM verifier or bonuses
        actual_score = test_output + 0.001 * env_interactions
        scores_per_test[current_input] = actual_score
        z = actual_score

      if current_input not in invalid_reasons and scores_per_test.get(current_input) == -1:
        invalid_reasons[current_input] = "reason_not_found"
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'status': 'evaluated',
        'version_generated': version_generated,
        'raw_sample': sample,
        'function_name': new_function.name,
        'function_body': new_function.body,
        "env_interactions": env_interactions,
        'ans': ans,
        'runs_ok': runs_ok,
        'args_ignored': _args_ignored,
        'invalid': any(v == -1 for v in scores_per_test.values()),
        'invalid_reasons': invalid_reasons if invalid_reasons else {},
        'island_id': island_id,
        'scores': scores_per_test
    }
    self._append_json_line(self._log_file, log_entry)
    
    if scores_per_test:
      self._database.register_program(new_function, island_id, scores_per_test)

    return z


