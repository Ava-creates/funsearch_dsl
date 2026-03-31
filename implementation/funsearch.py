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
import os
import re
import ast
import glob
import math
import argparse
from datetime import datetime
from collections.abc import Sequence
from typing import Any
import textwrap
import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from funsearch.implementation import code_manipulation
from funsearch.implementation import config as config_lib
from funsearch.implementation import evaluator
from funsearch.implementation import programs_database
from funsearch.implementation import sampler
from src.pipeline.grid_generation import ensure_function_grid_spec
try:
    from vllm import LLM as vLLM
except Exception:
    vLLM = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global lock for thread-safe vLLM access when using shared instances
_vllm_lock = threading.Lock()


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

    def __init__(self, model_type: str = 'huggingface', model_path: str = "/scratch/avani/qwen", shared_vllm=None):
        """Initialize FunSearch with model configuration.
        Args:
            model_type: Type of model to use ('huggingface' or 'ollama')
            model_path: Path to the model (for huggingface models)
            shared_vllm: Optional shared vLLM instance (for compatibility, but not used in sequential mode)
        """
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.shared_vllm = shared_vllm  # Store for passing to evaluators/samplers
        # self._initialize_model()
        self.vllm_lock = _vllm_lock  # Thread lock for safe concurrent access to shared vLLM

    def _initialize_shared_vllm(self):
        """Initialize shared vLLM instance for sampler and evaluator."""
        if self.model_type == "huggingface" and self.shared_vllm is None:
            if vLLM is None:
                print("Failed to initialize shared vLLM: vLLM not available")
                self.shared_vllm = None
                return
            try:
                self.shared_vllm = vLLM(
                    model="/scratch/avani/gpt", 
                    tensor_parallel_size=4,
                    gpu_memory_utilization=0.6  # Reduced to 60% to handle parallel jobs
                )
                print("Initialized shared vLLM instance for sampler and evaluator (using reduced memory settings)")
            except Exception as e:
                print(f"Failed to initialize shared vLLM: {e}")
                self.shared_vllm = None

    def _initialize_model(self):
        """Initialize the model and tokenizer based on model_type."""
        if self.model_type == 'huggingface':
            # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            # self.model = AutoModelForCausalLM.from_pretrained(
            #     self.model_path,
            #     torch_dtype=torch.float16,
            #     device_map="auto")
            self.model = None
            self.tokenizer = None

        else:
            self.model = None
            self.tokenizer = None

    def _read_function_from_file(self, function_name: str) -> str:
        """Read function implementation from a file named after the function.
        Args:
            function_name: Name of the function to read
            
        Returns:
            The function implementation as a string
        """
        try:
            # function_name = "function_specific_prompts/"+ function_name
            with open(function_name, 'r',  encoding='utf-8') as f:
                content = f.read()
                function_body = content
                return function_body
        except FileNotFoundError:
            return "return []"

    def _init_function_from_file(self, function_name: str) -> str:
        """Read function initialization from a file named after the function.
        
        Args:
            function_name: Name of the function to read
            
        Returns:
            The function initialization as a string
        """
        try:
            function_name = function_name.lower()

            with open(function_name, 'r',  encoding='utf-8') as f:
                content = f.read()
                function_body = content
                return function_body
        except FileNotFoundError:
            return "return []"

    def _extract_grid_paths(self, specification: str, spec_file: str | None) -> list[str]:
        """Extract JSON grid spec paths from the specification text."""
        paths = re.findall(r"['\"]([^'\"]+\\.json)['\"]", specification or "")
        if not paths:
            return []
        base_dir = os.path.dirname(spec_file) if spec_file else ""
        normalized = []
        for p in paths:
            if os.path.isabs(p):
                normalized.append(p)
            elif base_dir:
                normalized.append(os.path.normpath(os.path.join(base_dir, p)))
            else:
                normalized.append(os.path.normpath(p))
        seen = set()
        deduped = []
        for p in normalized:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        return deduped

    def _extract_function_metadata(self, specification: str, function_to_evolve: str) -> tuple[str, str]:
        """Get function docstring (first line) and argument list (excluding env)."""
        description = ""
        arg_list = ""
        try:
            tree = ast.parse(specification or "")
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == function_to_evolve:
                    raw_doc = ast.get_docstring(node) or ""
                    description = raw_doc.strip().split("\n")[0] if raw_doc else ""
                    args = [a.arg for a in node.args.args if a.arg not in ("self", "env")]
                    arg_list = ", ".join(args)
                    break
        except Exception as e:
            print(f"Warning: could not parse specification for metadata: {e}")
        return description, arg_list

    def _regenerate_grids_if_needed(
        self,
        specification: str,
        spec_file: str | None,
        function_to_evolve: str,
        max_attempts: int,
        experiment_dir: str | None = None,
    ) -> bool:
        """Attempt to regenerate grid specs when initial pass_check fails."""
        max_attempts = max(0, int(max_attempts))
        if max_attempts <= 0:
            print("[grid regen] Skipping: max_attempts<=0")
            return False

        # Ensure vLLM is available for grid generation
        if self.shared_vllm is None and self.model_type == "huggingface":
            self._initialize_shared_vllm()
        if self.shared_vllm is None:
            print("[grid regen] Skipping: shared vLLM unavailable")
            return False

        grid_paths = self._extract_grid_paths(specification, spec_file)
        if not grid_paths and experiment_dir:
            grids_dir = os.path.join(experiment_dir, "grids")
            os.makedirs(grids_dir, exist_ok=True)
            base_names = {function_to_evolve, function_to_evolve.lower()}
            discovered_paths: list[str] = []
            for base_name in base_names:
                discovered_paths.extend(
                    glob.glob(os.path.join(grids_dir, f"{base_name}_dsl*_case*.json"))
                )
                discovered_paths.extend(
                    glob.glob(os.path.join(grids_dir, f"{base_name}_case*.json"))
                )
            # Deduplicate while preserving deterministic order.
            grid_paths = sorted(set(discovered_paths))

        if not grid_paths:
            print("[grid regen] No grid paths found in specification")
            return False

        description, arg_list = self._extract_function_metadata(specification, function_to_evolve)
        if not description:
            description = function_to_evolve
        if not arg_list:
            arg_list = "None"

        def _maybe_read(path: str) -> str:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                return ""

        env_description = _maybe_read("prompt_specifications/nld.txt")
        recipes_text = _maybe_read("craft/resources/recipes.yaml")

        regenerated_any = False
        for attempt in range(max_attempts):
            for path in grid_paths:
                try:
                    dirpath = os.path.dirname(path)
                    if dirpath:
                        os.makedirs(dirpath, exist_ok=True)
                    ensure_function_grid_spec(
                        func_name=function_to_evolve,
                        description=description,
                        recipes_path="craft/resources/recipes.yaml",
                        output_path=path,
                        shared_vllm=self.shared_vllm,
                        default_task_name=None,
                        prompt_path="prompt_specifications/grid_prompt_old.txt",
                        func_args=arg_list,
                        env_description=env_description,
                        recipes_text=recipes_text,
                        attempts=5,
                    )
                    regenerated_any = True
                    print(f"[grid regen] Attempt {attempt + 1}/{max_attempts}: regenerated grid {path}")
                except Exception as e:
                    print(f"[grid regen] Attempt {attempt + 1}/{max_attempts}: failed to regenerate {path}: {e}")
        return regenerated_any

    def _replace_function_in_specification(self, specification: str, function_name: str, function_init:str) -> str:
        """Replace or add a function in the specification.
        
        Args:
            specification: The original specification string
            function_name: Name of the function to replace/add
            
        Returns:
            Modified specification string
        """
 # print(function_name)
        function_body = self._read_function_from_file(function_name).strip()
        # print(function_body)
        function_init_content = self._init_function_from_file(function_init)
        # print("function_init", function_init_content)
        
        # Extract function body after the colon
        if ":" in function_init_content:
            function_init_body = function_init_content[function_init_content.index(":")+1:]
        else:
            function_init_body = function_init_content
        
        # Split into lines and process each line
        lines = function_init_body.split('\n')
        
        # Remove leading and trailing empty lines
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            indented_function = "  return []"
        else:
            # Find minimum indentation of non-empty lines
            min_indent = float('inf')
            for line in lines:
                if line.strip():  # Only consider non-empty lines
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)
            
            if min_indent == float('inf'):
                min_indent = 0
            
            # Remove minimum indentation from all lines while preserving relative structure
            dedented_lines = []
            for line in lines:
                if line.strip():
                    # Remove exactly min_indent characters from the start
                    # This preserves relative indentation for nested structures
                    if len(line) >= min_indent:
                        dedented_line = line[min_indent:]
                    else:
                        dedented_line = line.lstrip()
                    dedented_lines.append(dedented_line)
                else:
                    # Preserve empty lines as-is
                    dedented_lines.append('')
            
            # Find the new minimum indentation after dedenting (for nested structures)
            new_min_indent = float('inf')
            for line in dedented_lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    new_min_indent = min(new_min_indent, indent)
            
            if new_min_indent == float('inf'):
                new_min_indent = 0
            
            # Normalize: remove any remaining base-level indentation, preserve relative indentation
            # If new_min_indent > 0, subtract it from all lines to normalize base to 0
            # If new_min_indent is 0, still strip small amounts of leading whitespace (1-2 spaces)
            # from base-level lines (inconsistent indentation), but preserve larger indentation
            # for nested structures
            normalized_lines = []
            for line in dedented_lines:
                if line.strip():
                    # Remove base-level indentation (new_min_indent), preserving relative structure
                    if new_min_indent > 0 and len(line) >= new_min_indent:
                        # Check if line starts with at least new_min_indent spaces
                        if line[:new_min_indent].isspace():
                            normalized_line = line[new_min_indent:]
                        else:
                            normalized_line = line.lstrip()
                    elif new_min_indent == 0:
                        # Even if min_indent is 0, some lines might have inconsistent small indentation (1 space)
                        # Strip only 1 space as it's likely inconsistent base-level indentation
                        # But preserve 2+ spaces as they're likely for nested structures
                        leading_spaces = len(line) - len(line.lstrip())
                        if leading_spaces == 1:
                            # Single space - likely inconsistent base-level, strip it
                            normalized_line = line.lstrip()
                        else:
                            # 0 spaces or 2+ spaces - preserve (0 = base level, 2+ = nested structure)
                            normalized_line = line
                    else:
                        normalized_line = line
                    normalized_lines.append(normalized_line)
                else:
                    normalized_lines.append('')
            
            # Now add consistent 2-space indentation to all non-empty lines (including comments)
            # All lines with content (code, comments, etc.) get 2-space base indentation
            indented_lines = []
            for line in normalized_lines:
                if line.strip():  # Line has content (code, comments, etc.)
                    # Add 2-space indentation to all lines with content
                    # Base-level lines should have no leading whitespace after normalization
                    # Nested lines will have leading whitespace which we preserve
                    indented_lines.append('  ' + line)
                else:
                    # Empty line - preserve as empty
                    indented_lines.append('')
            
            indented_function = '\n'.join(indented_lines)
        
        # print("indented_function"+indented_function)
        return specification + "\n" + function_body + "\n" + indented_function

    def run(
        self,
        specification: str,
        inputs: Sequence[Any],
        config: config_lib.Config,
        function_to_implement,
        function_init,
        spec_file,
        experiment_dir: str = None,
        grid_lookup_experiment_dir: str | None = None,
    ):
        """Run the FunSearch experiment.
        Args:
            specification: The specification string containing the functions to evolve and run
            inputs: Sequence of inputs to test the evolved function
            config: Configuration for the experiment
            function_to_implement: Name of the function to implement
            function_init: Path to function initialization file
            spec_file: Path to specification file
            experiment_dir: Optional directory for experiment outputs/logs
            grid_lookup_experiment_dir: Optional experiment root used to find/regenerate grids
        """
        specification = self._replace_function_in_specification(specification, function_to_implement, function_init)
        function_to_evolve, function_to_run = _extract_function_names(specification)
        template = code_manipulation.text_to_program(specification)
        database = programs_database.ProgramsDatabase(
            config.programs_database, template, function_to_evolve)
            
        # Only initialize shared vLLM if not provided and model type is huggingface
        # Note: If shared_vllm is None, we try to create one. However, in stages that explicitly
        # want to share an instance (like stage_implement_cfg_single), they should create it first
        # and pass it here to ensure only ONE instance is created and shared.
        if self.model_type == "huggingface" and self.shared_vllm is None:
            self._initialize_shared_vllm()
        # Create a shared log file path for all evaluators to use
        shared_log_file = None
        if experiment_dir:
            results_dir = experiment_dir
            os.makedirs(results_dir, exist_ok=True)
            
            current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            # Sanitize function name and init for filename
            safe_function_name = os.path.basename(function_to_implement).replace("/", ":").replace("\\", ":")
            safe_function_init = os.path.basename(function_init).replace("/", ":").replace("\\", ":")
            safe_spec = os.path.basename(spec_file) if spec_file else "specification"
            safe_spec = safe_spec.replace("/", "").replace("\\", "")
            model_name = "huggingface" if self.model_type == "huggingface" else self.model_type
            shared_log_file = os.path.join(results_dir, f'{model_name}_q2.5_{safe_function_name}_{safe_function_init}_{safe_spec}_{current_date}.log')

        evaluators = []
        for _ in range(config.num_evaluators):
            evaluators.append(evaluator.Evaluator(
                database,
                template,
                self.model_type,
                function_to_evolve,
                function_to_run,
                inputs,
                function_init,
                spec_file,
                function_name=function_to_implement,
                experiment_dir=experiment_dir,
                shared_vllm=self.shared_vllm,
                vllm_lock=self.vllm_lock,  # Use lock for thread-safe access to shared vLLM
                results_tracker=getattr(self, 'results_tracker', None),
                log_file=shared_log_file
            ))

        print("specification", specification)
        # We send the initial implementation to be analysed by one of the evaluators.
        initial = template.get_function(function_to_evolve).body
        check = evaluators[0].analyse(initial, island_id=None, version_generated=None)

        # Regenerate grids when the initial implementation fails hard.
        while check == -1:
            regen_attempts = getattr(config, "grid_regeneration_attempts", 0)
            print(f"[init check] Initial implementation failed; regenerating grids (up to {regen_attempts} attempts)")
            for attempt in range(max(0, regen_attempts)):
                regenerated = self._regenerate_grids_if_needed(
                    specification,
                    spec_file,
                    function_to_evolve,
                    max_attempts=15,
                    experiment_dir=(
                        grid_lookup_experiment_dir
                        if grid_lookup_experiment_dir is not None
                        else experiment_dir
                    ),
                )
                if regenerated:
                    check = evaluators[0].analyse(initial, island_id=None, version_generated=None)
                    if check != -1:
                        break

        # Calculate number of iterations per sampler
        # If total_samples is set, calculate iterations to achieve that total
        # Otherwise use num_iterations
        if check >=1500:
            return
        if config.total_samples is not None:
            # Calculate iterations: total_samples / (num_samplers * samples_per_prompt)
            # Each sampler will run this many iterations internally
            num_iterations = math.ceil(config.total_samples / (config.num_samplers * config.samples_per_prompt))
            print(f"Target: {config.total_samples} total samples")
            print(f"  With {config.num_samplers} samplers and {config.samples_per_prompt} samples per prompt")
            print(f"  Each sampler will run {num_iterations} iterations (sequential, preserving evolution)")
            print(f"  Total samples will be: {config.num_samplers * num_iterations * config.samples_per_prompt}")
        else:
            num_iterations = config.num_iterations
            print(f"Using {num_iterations} iterations per sampler (total_samples not set)")

        # Create samplers - each sampler will run all iterations internally
        samplers = [sampler.Sampler(database, evaluators, config.samples_per_prompt,
        tokenizer=self.tokenizer, model_type=self.model_type, shared_vllm=self.shared_vllm, vllm_lock=self.vllm_lock,
        num_iterations=num_iterations)  # Each sampler runs all iterations internally
                    for _ in range(config.num_samplers)]

        # Run samplers sequentially (no parallelization)
        # Each sampler will complete all its iterations before the next sampler starts
        print(f"Running {config.num_samplers} samplers sequentially (each doing {num_iterations} iterations)...")
        for i, s in enumerate(samplers):
            print(f"  Sampler {i + 1}/{config.num_samplers} running {num_iterations} iterations...")
            s.sample()
            print(f"  Sampler {i + 1}/{config.num_samplers} completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_file', type=str, required=True, help='Path to specification file')
    parser.add_argument('--func_file', type=str, required=True, help='Func txt file')
    parser.add_argument('--func_init', type=str, required=True, help='the file with start implementatino of the function')
    parser.add_argument('--model_type', type=str, choices=['huggingface', 'ollama', "gemini"],
                       default='huggingface', help='Choose between huggingface or ollama models')
    args = parser.parse_args()

    with open(args.spec_file, 'r',  encoding='utf-8') as f:
        specification = f.read()
    inputs = [3]
    config = config_lib.Config()

    funsearch = FunSearch(model_type=args.model_type)
    funsearch.run(specification, inputs, config, args.func_file, args.func_init, args.spec_file)