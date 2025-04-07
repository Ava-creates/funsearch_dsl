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

"""A single-threaded implementation of the FunSearch pipeline."""
from collections.abc import Sequence
from typing import Any
import textwrap
from funsearch_dsl.implementation import code_manipulation
from funsearch_dsl.implementation import config as config_lib
from funsearch_dsl.implementation import evaluator
from funsearch_dsl.implementation import programs_database
from funsearch_dsl.implementation import sampler



# def evolve(func):
#     def wrapper(*args, **kwargs):
#         print("Running decorated function...")
#         return func(*args, **kwargs)
#     return wrapper
# def run(func):
#     return func

# def evolve(func):
#     return func



def _extract_function_names(specification: str) -> tuple[str, str]:
  """Returns the name of the function to evolve and of the function to run."""
  # run_functions = list(
  #     code_manipulation.yield_decorated(specification, 'funsearch', 'run'))
  # if len(run_functions) != 1:
  #   raise ValueError('Expected 1 function decorated with `@funsearch.run`.')
  # evolve_functions = list(
  #     code_manipulation.yield_decorated(specification, 'funsearch', 'evolve'))
  # if len(evolve_functions) != 1:
  #   raise ValueError('Expected 1 function decorated with `@funsearch.evolve`.')
  return  "priority","evaluate"


def main(specification: str, inputs: Sequence[Any], config: config_lib.Config):
  """Launches a FunSearch experiment."""
  function_to_evolve, function_to_run = _extract_function_names(specification)

  template = code_manipulation.text_to_program(specification)
  print(template)
  print("template end")
  database = programs_database.ProgramsDatabase(
      config.programs_database, template, function_to_evolve)

  evaluators = []
  for _ in range(config.num_evaluators):
    evaluators.append(evaluator.Evaluator(
        database,
        template,
        function_to_evolve,
        function_to_run,
        inputs,
    ))
  # We send the initial implementation to be analysed by one of the evaluators.
  initial = template.get_function(function_to_evolve).body
  evaluators[0].analyse(initial, island_id=None, version_generated=None)

  samplers = [sampler.Sampler(database, evaluators, config.samples_per_prompt)
              for _ in range(config.num_samplers)]

  # This loop can be executed in parallel on remote sampler machines. As each
  # sampler enters an infinite loop, without parallelization only the first
  # sampler will do any work.
  for s in samplers:
    s.sample()

if __name__ == "__main__":
    # Define your specification string
    specification = textwrap.dedent('''
    """Finds large cap sets."""
    import itertools
    import numpy as np
 
    # @funsearch.run
    def evaluate(n: int) -> int:
      capset = solve(n)

      return 20

    def solve(n: int) -> np.ndarray:
      all_vectors = np.array(list(itertools.product((0, 1, 2), repeat=n)), dtype=np.int32)

      powers = 3 ** np.arange(n - 1, -1, -1)
      priorities = np.array([priority(tuple(vector), n) for vector in all_vectors])

      capset = np.empty(shape=(0, n), dtype=np.int32)
      while np.any(priorities != -np.inf):
            # print("in loop")
            max_index = np.argmax(priorities)
            vector = all_vectors[None, max_index]  # [1, n]
            blocking = np.einsum('cn,n->c', (- capset - vector) % 3, powers)  # [C]
            priorities[blocking] = -np.inf
            priorities[max_index] = -np.inf
            capset = np.concatenate([capset, vector], axis=0)

      return 4

    # @funsearch.evolve
    def priority(el: tuple[int, ...], n: int) -> float:
        return 0.0
''')
    # Define your inputs (adjust these as needed)
    inputs = [3]
    config = config_lib.Config()

    # Call main with the defined arguments.
    main(specification, inputs, config)