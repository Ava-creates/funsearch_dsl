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

from funsearch.implementation import code_manipulation
from funsearch.implementation import programs_database


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


def _trim_function_body(generated_code: str) -> str:
  """Extracts the body of the generated function, trimming anything after it."""
  if not generated_code:
    return ''
  code = f'def fake_function_header():\n{generated_code}'
  tree = None
  # We keep trying and deleting code from the end until the parser succeeds.
  while tree is None:
    try:
      tree = ast.parse(code)
    except SyntaxError as e:
      code = '\n'.join(code.splitlines()[:e.lineno - 1])
  if not code:
    # Nothing could be saved from `generated_code`
    return ''

  visitor = _FunctionLineVisitor('fake_function_header')
  visitor.visit(tree)
  body_lines = code.splitlines()[1:visitor.function_end_line]
  return '\n'.join(body_lines) + '\n\n'


def _sample_to_program(
    generated_code: str,
    version_generated: int | None,
    template: code_manipulation.Program,
    function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
  """Returns the compiled generated function and the full runnable program."""

  body = _trim_function_body(generated_code)

  if version_generated is not None:
    body = code_manipulation.rename_function_calls(
        body,
        f'{function_to_evolve}_v{version_generated}',
        function_to_evolve)

  program = copy.deepcopy(template)
  evolved_function = program.get_function(function_to_evolve)
  evolved_function.body = body
  return evolved_function, str(program)


class Sandbox:
  """Sandbox for executing generated code."""

  def run(
        self,
        program: str,
        function_to_run: str,
        test_input: Any,
        timeout_seconds: int
    ) -> Tuple[Any, bool]:
            """
            Executes Python code in a subprocess and returns:
            - The function's output
            - Boolean indicating successful execution
            """
            # Create unique filename using process ID and timestamp
            temp_dir = os.getcwd()
            unique_id = f"{os.getpid()}_{int(time.time() * 1000000)}"
            script_path = f'generated_code_{unique_id}.py'
            script_path = os.path.join(temp_dir, script_path)

            
            # Create complete executable program
            full_program = f"""
{program} 
print({function_to_run}())
          """
            
            try:
                with open(script_path, 'w') as f:
                    f.write(full_program.strip())

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
                    errors='replace'
                )

                output = result.stdout.strip()
                # Convert numpy types to native Python types and parse as JSON
                output = output.replace("np.float64", "")
                output = output.replace("np.float32", "")
                output = output.replace("(", "").replace(")", "")
                result = ast.literal_eval(output)

                # Access the values
                output, actions_count = result[0], result[1]
                print("output ", output)
                
                # output = result.stdout.strip()
                # result = ast.literal_eval(output)
                # print("result ", result)
                # # Access the values
                # output, actions_count = result
                # Try to parse numerical output
                # output = result.stdout.strip()
                # print("output ", output)
                try:
                    return float(output), True , actions_count
                except ValueError:
                    return -1, True, 0
                    
            except subprocess.TimeoutExpired:
                return -1, False, 0
            except subprocess.CalledProcessError as e:
                print(f"Process Error: Command failed with exit code {e.returncode}")
                print(f"Command: {e.cmd}")
                print(f"Output: {e.stdout}")
                print(f"Error: {e.stderr}")
                return -1, False, 0
            except Exception as e:
                print(f"Unexpected Error: {e}")
                return -1, False, 0 
            finally:
                # Clean up the temporary file
                if os.path.exists(script_path):
                    os.remove(script_path)


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
      function_name
  ):
    self._database = database
    self._template = template
    self._function_to_evolve = function_to_evolve
    self._function_to_run = function_to_run
    self._inputs = inputs
    self._timeout_seconds = 300 #100 sec for each env difficulty
    self._function_name = function_name
    self._sandbox = Sandbox()
    self._log_file = None  # Will be initialized on first use
    self._model_name = model_name
    self._function_init = function_to_init
    self.specification = specification

  def analyse(
      self,
      sample: str,
      island_id: int | None,
      version_generated: int | None,
  ) -> None:
    """Compiles the sample into a program and executes it on test inputs."""
    # print("sample from the llm", sample)
    new_function, program = _sample_to_program(
        sample, version_generated, self._template, self._function_to_evolve)
    print("program ", program)
    if new_function == "":
      return 0
    scores_per_test = {}
    print("function to run", self._function_to_run)
    for current_input in self._inputs:
      test_output, runs_ok, env_interactions = self._sandbox.run(
          program, self._function_to_run, current_input, self._timeout_seconds)
      print("runs_ok:", runs_ok)
      print("test_output", test_output)
      print("env_interactions", env_interactions)
      if(runs_ok == False):
        scores_per_test[current_input] = -1
      if (runs_ok and not _calls_ancestor(program, self._function_to_evolve)
          and test_output is not None):
        if not isinstance(test_output, (int, float)):
          scores_per_test[current_input] = -1
          # raise ValueError('@function.run did not return an int/float score.')
        scores_per_test[current_input] = test_output
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'function_name': new_function.name,
        'function_body': new_function.body,
        "env_interactions": env_interactions,
        'island_id': island_id,
        'scores': scores_per_test
    }
    
    if self._log_file is None:
        results_dir = 'results/funsearch'
        os.makedirs(results_dir, exist_ok=True)
        
        current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self._log_file = os.path.join(results_dir, f'{self._model_name}_q2.5_{self._function_name.replace("/",":")}_{self._function_init.replace("/",":")}_{self.specification.replace("/", "")}_{current_date}.log')
    
    with open(self._log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    if scores_per_test:
      self._database.register_program(new_function, island_id, scores_per_test)

    if(self._function_name == "move" and sum(scores_per_test.values()) == 4):
        return 1
    elif(self._function_name == "use" and sum(scores_per_test.values()) == 2):
        return 1
    elif( "craft" in self._function_name and sum(scores_per_test.values())  >= 1):
        return 1
    elif(self._function_name == "collect" and sum(scores_per_test.values())  == 1.5 ):
        return 1 
    else:
      return 0 


