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
from typing import Any, List
import textwrap
from funsearch.implementation import code_manipulation
from funsearch.implementation import config as config_lib
from funsearch.implementation import evaluator
from funsearch.implementation import programs_database
from funsearch.implementation import sampler
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import argparse
import re
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _extract_function_names(specification: str) -> tuple[str, str]:
  """Returns the name of the function to evolve and of the function to run."""
  run_functions = list(
      code_manipulation.yield_decorated(specification, 'funsearch', 'run'))
  if len(run_functions) != 1:
    raise ValueError('Expected 1 function decorated with `@funsearch.run`.')
  evolve_functions = list(
      code_manipulation.yield_decorated(specification, 'funsearch', 'evolve'))
  if len(evolve_functions) != 1:
    raise ValueError('Expected 1 function decorated with `@funsearch.evolve`.')
  return evolve_functions[0], run_functions[0]



class FunSearch:
    """A class that implements the FunSearch pipeline."""
    
    def __init__(self, model_type: str = 'huggingface', model_path: str = "/scratch/avani/qwen"):
        """Initialize FunSearch with model configuration.
        
        Args:
            model_type: Type of model to use ('huggingface' or 'ollama')
            model_path: Path to the model (for huggingface models)
        """
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        # self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model and tokenizer based on model_type."""
        if self.model_type == 'huggingface':
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:  
            self.model = None
            self.tokenizer = None

    def _read_function_from_file(self, function_name: str) -> tuple[str, str]:
        """Read function implementation and function to run from a file named after the function.
        
        Args:
            function_name: Name of the function to read
            
        Returns:
            A tuple containing:
            - The function implementation as a string
            - The function to run as a string
        """
        try:
            with open(f"{function_name}.txt", 'r') as f:
                content = f.read()
                function_body = content
                function_to_run = function_name
                return function_body
        except FileNotFoundError:
            return "return []"

    def _init_function_from_file(self, function_name: str) -> tuple[str, str]:
        """Read function implementation and function to run from a file named after the function.
        
        Args:
            function_name: Name of the function to read
            
        Returns:
            A tuple containing:
            - The function implementation as a string
            - The function to run as a string
        """
        try:
            function_name = function_name.lower()
            print(function_name)
            with open(function_name, 'r') as f:
                content = f.read()
                function_body = content
                function_to_run = function_name
                return function_body
        except FileNotFoundError:
            return "return []"

    def _replace_function_in_specification(self, specification: str, function_name: str, function_init:str) -> str:
        """Replace or add a function in the specification.
        
        Args:
            specification: The original specification string
            function_name: Name of the function to replace/add
            
        Returns:
            Modified specification string
        """
        # Read function implementation from file
        function_body = self._read_function_from_file(function_name)
        
        function_init = self._init_function_from_file(function_init)
        function_init = function_init[function_init.index("def craft(env, item):")+21:]
        dedented = textwrap.dedent(function_init)
        indented_function = textwrap.indent(dedented, '  ')
        return specification + "\n" + function_body + "\n" +indented_function

    def run(self, specification: str, inputs: Sequence[Any], config: config_lib.Config, function_to_implement: str = None, function_init: str = None):
        """Run the FunSearch experiment.
        
        Args:
            specification: The specification string containing the functions to evolve and run
            inputs: Sequence of inputs to test the evolved function
            config: Configuration for the experiment
            function_to_implement: Name of the function to implement
        """
        
        specification = self._replace_function_in_specification(specification, function_to_implement, function_init)
        print(specification)
        function_to_evolve, function_to_run = _extract_function_names(specification)
        template = code_manipulation.text_to_program(specification)
        database = programs_database.ProgramsDatabase(
            config.programs_database, template, function_to_evolve)

        evaluators = []
        for _ in range(config.num_evaluators):
            evaluators.append(evaluator.Evaluator(
                database,
                template,
                self.model_type,
                function_to_evolve,
                function_to_run,
                inputs,
                function_name=function_to_implement
            ))
        # We send the initial implementation to be analysed by one of the evaluators.
        initial = template.get_function(function_to_evolve).body
        evaluators[0].analyse(initial, island_id=None, version_generated=None)

        samplers = [sampler.Sampler(database, evaluators, config.samples_per_prompt, 
                             tokenizer=self.tokenizer, model_type=self.model_type)
                    for _ in range(config.num_samplers)]

        for s in samplers:
            s.sample()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_file', type=str, required=True, help='Path to specification file')
    parser.add_argument('--function', type=str, required=True, help='Name of function to implement')
    parser.add_argument('--function_init', type=str, required=True, help='the file with start implementatino of the function')
    parser.add_argument('--model_type', type=str, choices=['huggingface', 'ollama', "gemini"], 
                       default='huggingface', help='Choose between huggingface or ollama models')
    args = parser.parse_args()

    with open(args.spec_file, 'r') as f:
        specification = f.read()
    
    inputs = [3]
    config = config_lib.Config()

    funsearch = FunSearch(model_type=args.model_type)
    funsearch.run(specification, inputs, config, args.function, args.function_init)
