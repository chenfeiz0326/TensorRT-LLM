#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import requests
import yaml


class BenchmarkRunner:

    def __init__(self,
                 output_folder: str,
                 config_file: str,
                 skip_pattern: str = None,
                 select_pattern: str = None):
        self.output_folder = Path(output_folder)
        self.config_file = Path(config_file)

        # Treat empty or "default" values as None (default behavior)
        self.skip_pattern = None if not skip_pattern or skip_pattern.lower(
        ) == "default" else skip_pattern
        self.select_pattern = None if not select_pattern or select_pattern.lower(
        ) == "default" else select_pattern

        self.skip_test_cases: Set[int] = set()
        self.skip_concurrencies: Dict[int, Set[int]] = {}
        self.select_test_cases: Set[int] = set()
        self.select_concurrencies: Dict[int, Set[int]] = {}

        if self.skip_pattern:
            self.parse_skip_pattern(self.skip_pattern)

        if self.select_pattern:
            self.parse_select_pattern(self.select_pattern)

        # Execution plan: {test_case_id: [concurrency_indices]}
        self.execution_plan: Dict[int, List[int]] = {}

        # Model path mapping
        llm_models_root = os.environ.get('LLM_MODELS_ROOT', '/home/scratch.trt_llm_data/llm-models')
        self.model_paths = {
            "70B-FP4":
            f"{llm_models_root}/llama-3.3-models/Llama-3.3-70B-Instruct-FP4",
            "70B-FP8":
            f"{llm_models_root}/llama-3.3-models/Llama-3.3-70B-Instruct-FP8",
            "Scout-FP4":
            f"{llm_models_root}/llama4-models/Llama-4-Scout-17B-16E-Instruct-FP4",
            "Scout-FP8":
            f"{llm_models_root}/llama4-models/Llama-4-Scout-17B-16E-Instruct-FP8",
            "R1-FP8":
            f"{llm_models_root}/DeepSeek-R1/DeepSeek-R1",
            "R1-FP4":
            f"{llm_models_root}/DeepSeek-R1/DeepSeek-R1-0528-FP4"
        }

        self.hf_model_paths = {
            "70B-FP4":  "meta-llama/Meta-Llama-3.3-70B-Instruct-FP4",
            "70B-FP8":  "meta-llama/Meta-Llama-3.3-70B-Instruct-FP8",
            "Scout-FP4": "meta-llama/Llama-4-Scout-17B-16E-Instruct-FP4",
            "Scout-FP8": "meta-llama/Llama-4-Scout-17B-16E-Instruct-FP8",
            "R1-FP8": "deepseek-ai/DeepSeek-R1",
            "R1-FP4": "deepseek-ai/DeepSeek-R1-0528-FP4"
        }

        # Set environment variables
        os.environ['TQDM_MININTERVAL'] = '1000'
        os.environ['PRINT_ITER_LOG'] = 'false'

        # Capture system information
        self.node_name = self.get_node_name()
        self.gpu_info = self.get_gpu_info()

        # Change to output directory
        os.chdir(self.output_folder)
        
        # Track failed test cases for reproduction script generation
        self.failed_test_cases: List[Dict[str, Any]] = []

    def get_node_name(self) -> str:
        """Get the current node name"""
        try:
            result = subprocess.run("hostname",
                                    shell=True,
                                    capture_output=True,
                                    text=True,
                                    check=True)
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    def get_gpu_info(self) -> str:
        """Get GPU information from nvidia-smi"""
        try:
            result = subprocess.run("nvidia-smi",
                                    shell=True,
                                    capture_output=True,
                                    text=True,
                                    check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"nvidia-smi failed with error code {e.returncode}\nError output: {e.stderr}"
        except FileNotFoundError:
            return "nvidia-smi not found"

    def parse_skip_pattern(self, skip_pattern: str) -> None:
        """Parse skip pattern like '2,4-1' to determine what to skip"""
        if not skip_pattern:
            return

        parts = skip_pattern.split(',')
        for part in parts:
            part = part.strip()
            if not part:  # Skip empty parts
                continue

            if '-' in part:
                # Format: "test_case-concurrency_index" (1-based)
                try:
                    test_case_str, concurrency_str = part.split('-')
                    test_case_id = int(test_case_str)
                    concurrency_index = int(
                        concurrency_str) - 1  # Convert to 0-based

                    if test_case_id not in self.skip_concurrencies:
                        self.skip_concurrencies[test_case_id] = set()
                    self.skip_concurrencies[test_case_id].add(concurrency_index)
                except ValueError:
                    raise ValueError(
                        f"Invalid skip pattern '{part}'. Expected format: 'test_case-concurrency_index' (e.g., '2-1')"
                    )
            else:
                # Format: "test_case" - skip entire test case
                try:
                    test_case_id = int(part)
                    self.skip_test_cases.add(test_case_id)
                except ValueError:
                    raise ValueError(
                        f"Invalid test case ID '{part}' in skip pattern. Must be a valid integer."
                    )

        print(f"Skipping test cases: {sorted(self.skip_test_cases)}")
        print(f"Skipping concurrencies: {self.skip_concurrencies}")

    def parse_select_pattern(self, select_pattern: str) -> None:
        """Parse select pattern like '1,3,5' or '1-1,2-3' to determine which test cases/concurrencies to run"""
        if not select_pattern:
            return

        self.select_concurrencies: Dict[int, Set[int]] = {}

        parts = select_pattern.split(',')
        for part in parts:
            part = part.strip()
            if not part:  # Skip empty parts
                continue

            if '-' in part:
                # Format: "test_case-concurrency_index" (1-based)
                try:
                    test_case_str, concurrency_str = part.split('-')
                    test_case_id = int(test_case_str)
                    concurrency_index = int(
                        concurrency_str) - 1  # Convert to 0-based

                    if test_case_id not in self.select_concurrencies:
                        self.select_concurrencies[test_case_id] = set()
                    self.select_concurrencies[test_case_id].add(
                        concurrency_index)
                except ValueError:
                    raise ValueError(
                        f"Invalid select pattern '{part}'. Expected format: 'test_case-concurrency_index' (e.g., '2-1')"
                    )
            else:
                # Format: "test_case" - select entire test case
                try:
                    test_case_id = int(part)
                    self.select_test_cases.add(test_case_id)
                except ValueError:
                    raise ValueError(
                        f"Invalid test case ID '{part}' in select pattern. Must be a valid integer."
                    )

        print(f"Selected test cases: {sorted(self.select_test_cases)}")
        print(f"Selected concurrencies: {self.select_concurrencies}")

    def build_execution_plan(self, test_cases: List[Dict[str, Any]]) -> None:
        """Build execution plan by analyzing config file, skip_pattern, and select_pattern"""
        self.execution_plan.clear()

        # Step 1: Initialize execution plan based on select_pattern
        if not self.select_pattern:
            # If select_pattern is empty or default, include all test cases with all concurrencies
            for test_case in test_cases:
                test_case_id = test_case['id']
                all_concurrencies = list(
                    range(len(test_case['concurrency_iterations'])))
                self.execution_plan[test_case_id] = all_concurrencies
        else:
            # If select_pattern is specified, only include selected test cases and concurrencies
            for test_case in test_cases:
                test_case_id = test_case['id']

                # Check if this test case is selected
                if test_case_id in self.select_test_cases:
                    # Test case is selected - include all concurrencies
                    all_concurrencies = list(
                        range(len(test_case['concurrency_iterations'])))
                    self.execution_plan[test_case_id] = all_concurrencies
                elif test_case_id in self.select_concurrencies:
                    # Specific concurrencies are selected for this test case
                    selected_concurrencies = list(
                        self.select_concurrencies[test_case_id])
                    # Validate that selected concurrencies exist in config
                    max_concurrency_index = len(
                        test_case['concurrency_iterations']) - 1
                    valid_concurrencies = [
                        c for c in selected_concurrencies
                        if 0 <= c <= max_concurrency_index
                    ]
                    if valid_concurrencies:
                        self.execution_plan[test_case_id] = valid_concurrencies

        # Step 2: Apply skip_pattern to remove test cases and concurrencies
        # Remove entire test cases that are in skip_test_cases
        for test_case_id in self.skip_test_cases:
            if test_case_id in self.execution_plan:
                del self.execution_plan[test_case_id]

        # Remove specific concurrencies that are in skip_concurrencies
        for test_case_id, skip_concurrency_indices in self.skip_concurrencies.items(
        ):
            if test_case_id in self.execution_plan:
                # Remove skipped concurrencies from the list
                remaining_concurrencies = [
                    c for c in self.execution_plan[test_case_id]
                    if c not in skip_concurrency_indices
                ]
                if remaining_concurrencies:
                    self.execution_plan[test_case_id] = remaining_concurrencies
                else:
                    # If no concurrencies remain, remove the entire test case
                    del self.execution_plan[test_case_id]

        # Step 3: Clean up - remove test cases with empty concurrency lists
        # (This should not happen with the above logic, but just to be safe)
        test_cases_to_remove = []
        for test_case_id, concurrencies in self.execution_plan.items():
            if not concurrencies:
                test_cases_to_remove.append(test_case_id)

        for test_case_id in test_cases_to_remove:
            del self.execution_plan[test_case_id]

    def print_execution_plan(self, test_cases: List[Dict[str, Any]]) -> None:
        """Print which test cases and concurrencies will be executed"""
        print("\n" + "=" * 80)
        print("EXECUTION PLAN")
        print("=" * 80)

        total_test_cases = 0
        total_concurrencies = 0

        for test_case in test_cases:
            test_case_id = test_case['id']
            model_label = test_case['model']

            # Check if this test case is in execution plan
            if test_case_id not in self.execution_plan:
                print(f"Test Case {test_case_id}: {model_label} - SKIPPED")
                continue

            total_test_cases += 1
            print(f"\nTest Case {test_case_id}: {model_label}")
            print(
                f"  Config: GPUs={test_case['gpus']}, TP={test_case['tp']}, EP={test_case['ep']}, attn_backend={test_case['attn_backend']}, moe_backend={test_case['moe_backend']}"
            )

            # Get concurrencies from execution plan
            concurrencies_to_run = []
            for concurrency_index in self.execution_plan[test_case_id]:
                concurrency, iteration = test_case['concurrency_iterations'][
                    concurrency_index]
                concurrencies_to_run.append(
                    (concurrency_index + 1, concurrency,
                     iteration))  # +1 for 1-based display
                total_concurrencies += 1

            print(
                f"  Concurrencies to run ({len(concurrencies_to_run)}/{len(test_case['concurrency_iterations'])}):"
            )
            for concurrency_num, concurrency, iteration in concurrencies_to_run:
                print(
                    f"    {concurrency_num}. Concurrency={concurrency}, Iteration={iteration}"
                )

        print("\n" + "=" * 80)
        print(
            f"SUMMARY: {total_test_cases} test cases, {total_concurrencies} concurrencies will be executed"
        )
        print("=" * 80 + "\n")

    def generate_extra_llm_api_config(self, test_case: Dict[str, Any]) -> str:
        """Generate extra-llm-api-config.yml content"""
        config_lines = [
            "print_iter_log: true",
            f"enable_attention_dp: {str(test_case['enable_attention_dp']).lower()}",
            "disable_overlap_scheduler: false",
            "stream_interval: 10",
            f"attn_backend: {test_case['attn_backend']}",
            "cuda_graph_config:",
            "  enable_padding: true",
            f"  max_batch_size: {test_case['max_batch_size']}",
            "kv_cache_config:",
            "  dtype: fp8",
            f"  free_gpu_memory_fraction: {test_case['free_gpu_mem_fraction']}",
            "  enable_block_reuse: false",
        ]

        # Add moe_config if moe_backend is specified
        if test_case['moe_backend']:
            config_lines.append("moe_config:")
            config_lines.append(f"  backend: {test_case['moe_backend']}")

            if test_case['moe_max_num_tokens']:
                config_lines.append(
                    f"  max_num_tokens: {test_case['moe_max_num_tokens']}")

        return "\n".join(config_lines)

    def wait_for_server(self,
                        server_pid: int,
                        server_log_filename: str,
                        max_attempts: int = 360) -> bool:
        """Wait for server to be ready"""
        print("Waiting for trtllm-serve to be ready...")

        for attempt in range(1, max_attempts + 1):
            # Check if server is still running
            try:
                os.kill(server_pid, 0)  # Check if process exists
            except OSError:
                print("Error: Server process has died")
                return False

            # Check server log for runtime errors
            if self.check_for_runtime_error(server_log_filename):
                print(
                    f"RuntimeError detected in server log: {server_log_filename}"
                )
                print("Killing server process due to runtime error")
                try:
                    subprocess.run(f"kill -9 {server_pid}",
                                   shell=True,
                                   check=False)
                    subprocess.run(f"wait {server_pid} 2>/dev/null || true",
                                   shell=True,
                                   check=False)
                except Exception as e:
                    print(f"Warning: Error killing server process: {e}")
                return False

            # Try to connect to server
            try:
                response = requests.get("http://localhost:8000/v1/models",
                                        timeout=5)
                if response.status_code == 200:
                    print(
                        f"Server is ready! HTTP status: {response.status_code}")
                    return True
            except requests.RequestException:
                pass

            print(
                f"Attempt {attempt}/{max_attempts}: Server not ready yet, waiting..."
            )
            time.sleep(10)

        print(
            f"Error: Server did not become ready after {max_attempts} attempts")
        return False

    def check_for_runtime_error(self, log_file_path: str) -> bool:
        """Check if RuntimeError exists in log file"""
        try:
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r') as f:
                    content = f.read()
                    if "RuntimeError" in content or "runtime error" in content or "illegal memory access" in content or "terminate called" in content:
                        return True
        except Exception as e:
            print(f"Warning: Could not read log file {log_file_path}: {e}")
        return False

    def run_benchmark(self, test_case: Dict[str, Any], concurrency: int,
                      iteration: int, model_path: str,
                      server_log_filename: str) -> bool:
        """Run a single benchmark with monitoring. Returns True if successful, False if should skip test case"""
        num_prompts = concurrency * iteration

        print(
            f'Running benchmark with concurrency: {concurrency}, iteration: {iteration}, num-prompts: {num_prompts}'
        )

        # Build benchmark command
        benchmark_cmd = [
            "python", "-m", "tensorrt_llm.serve.scripts.benchmark_serving",
            "--model", model_path, "--dataset-name", "random", "--random-ids",
            "--num-prompts",
            str(num_prompts), "--random-input-len",
            str(test_case['isl']), "--random-output-len",
            str(test_case['osl']), "--random-range-ratio", "0.0",
            "--ignore-eos", "--percentile-metrics", "ttft,tpot,itl,e2el",
            "--max-concurrency",
            str(concurrency)
        ]

        print(f'Running benchmark with command:')
        print(' '.join(benchmark_cmd))
        print()

        # Prepare log filename
        benchmark_log_filename = (
            f"serve.{test_case['model']}.tp{test_case['tp']}.ep{test_case['ep']}."
            f"attn{test_case['attn_backend']}.moe{test_case['moe_backend']}."
            f"gpu{test_case['free_gpu_mem_fraction']}.batch{test_case['max_batch_size']}."
            f"isl{test_case['isl']}.osl{test_case['osl']}."
            f"tokens{test_case['max_num_tokens']}.moetokens{test_case['moe_max_num_tokens']}."
            f"concurrency{concurrency}.iter{iteration}.log")

        try:
            with open(benchmark_log_filename, 'w') as f:
                f.write(f"GPU Info: {self.gpu_info}\n")

            # Start benchmark as subprocess
            with open(benchmark_log_filename, 'a') as log_file:
                benchmark_process = subprocess.Popen(benchmark_cmd,
                                                     stdout=log_file,
                                                     stderr=subprocess.STDOUT)

            # Monitor logs every 60 seconds with timeout
            print(
                f"Starting log monitoring for benchmark process (PID: {benchmark_process.pid})"
            )

            start_time = time.time()
            timeout_seconds = 3600  # 1 hour timeout

            while benchmark_process.poll() is None:  # Process is still running
                time.sleep(60)  # Wait 60 seconds

                # Check if benchmark has been running for more than 1 hour
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout_seconds:
                    print(
                        f"Benchmark timeout after {elapsed_time:.0f} seconds (>{timeout_seconds} seconds)"
                    )
                    print("Killing benchmark process due to timeout")
                    try:
                        subprocess.run(f"kill -9 {benchmark_process.pid}",
                                       shell=True,
                                       check=False)
                        benchmark_process.wait(timeout=10)
                    except Exception as e:
                        print(f"Warning: Error killing benchmark process: {e}")
                    return False  # Signal to skip test case

                print(
                    f"Checking logs for RuntimeError... (benchmark PID: {benchmark_process.pid}, elapsed: {elapsed_time:.0f}s)"
                )

                # Check server log for RuntimeError
                if self.check_for_runtime_error(server_log_filename):
                    print(
                        f"RuntimeError found in server log: {server_log_filename}"
                    )
                    print(
                        "Killing benchmark process and skipping this test case")
                    try:
                        subprocess.run(f"kill -9 {benchmark_process.pid}",
                                       shell=True,
                                       check=False)
                        benchmark_process.wait(timeout=10)
                    except Exception as e:
                        print(f"Warning: Error killing benchmark process: {e}")
                    return False  # Signal to skip test case

                # Check benchmark log for RuntimeError
                if self.check_for_runtime_error(benchmark_log_filename):
                    print(
                        f"RuntimeError found in benchmark log: {benchmark_log_filename}"
                    )
                    print(
                        "Killing benchmark process and skipping this test case")
                    try:
                        subprocess.run(f"kill -9 {benchmark_process.pid}",
                                       shell=True,
                                       check=False)
                        benchmark_process.wait(timeout=10)
                    except Exception as e:
                        print(f"Warning: Error killing benchmark process: {e}")
                    return False  # Signal to skip test case

            # Process completed, check final return code
            return_code = benchmark_process.returncode
            if return_code != 0:
                print(
                    f"Benchmark process completed with error code: {return_code}"
                )

                # Read and display error output
                try:
                    with open(benchmark_log_filename, 'r') as f:
                        error_content = f.read()
                        print(
                            f"Benchmark error output:\n{error_content[-1000:]}"
                        )  # Last 1000 chars
                except Exception as e:
                    print(f"Could not read benchmark log: {e}")

                print(
                    f"Skipping this concurrency level and continuing with next one..."
                )
                print("-----------------------------------------")
                return True  # Continue with next concurrency, don't skip test case

            # Success case
            print(
                f"Benchmark completed successfully (PID: {benchmark_process.pid})"
            )

            # Add configuration summary to log file
            config_summary = (
                f"Completed benchmark with Configuration: "
                f"model_label={test_case['model']}, GPUs={test_case['gpus']}, "
                f"TP={test_case['tp']}, EP={test_case['ep']}, "
                f"attn_backend={test_case['attn_backend']}, "
                f"moe_backend={test_case['moe_backend']}, "
                f"enable_attention_dp={test_case['enable_attention_dp']}, "
                f"free_gpu_mem_fraction={test_case['free_gpu_mem_fraction']}, "
                f"max_batch_size={test_case['max_batch_size']}, "
                f"ISL={test_case['isl']}, OSL={test_case['osl']}, "
                f"max_num_tokens={test_case['max_num_tokens']}, "
                f"moe_max_num_tokens={test_case['moe_max_num_tokens']}, "
                f"Concurrency={concurrency}")
            with open(benchmark_log_filename, 'a') as f:
                f.write(f"\n{config_summary}\n")

            print("-----------------------------------------")
            return True  # Continue with next concurrency

        except Exception as e:
            print(
                f"Error running benchmark with concurrency {concurrency}: {e}")
            print(
                f"Skipping this concurrency level and continuing with next one..."
            )
            print("-----------------------------------------")
            return True  # Continue with next concurrency, don't skip test case

    def run_test_case(self, test_case: Dict[str, Any]) -> None:
        """Run a test case using the execution plan with retry logic"""
        model_label = test_case['model']
        test_case_id = test_case['id']
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            retry_count += 1
            print(f"Attempt {retry_count}/{max_retries} for test case {test_case_id} ({model_label})")
            
            try:
                success = self._run_test_case_attempt(test_case)
                if success:
                    print(f"Test case {test_case_id} ({model_label}) completed successfully on attempt {retry_count}")
                    return
                else:
                    print(f"Test case {test_case_id} ({model_label}) failed on attempt {retry_count}")
                    if retry_count < max_retries:
                        print(f"Retrying in 5 seconds...")
                        time.sleep(5)
                    else:
                        print(f"Test case {test_case_id} ({model_label}) failed after {max_retries} attempts. Skipping.")
                        self.add_failed_test_case(test_case, f"Failed after {max_retries} attempts - all retries exhausted")
                        return
            except Exception as e:
                print(f"Test case {test_case_id} ({model_label}) encountered exception on attempt {retry_count}: {e}")
                if retry_count < max_retries:
                    print(f"Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"Test case {test_case_id} ({model_label}) failed after {max_retries} attempts due to exceptions. Skipping.")
                    self.add_failed_test_case(test_case, f"Failed after {max_retries} attempts due to exception: {e} - all retries exhausted")
                    return

    def _run_test_case_attempt(self, test_case: Dict[str, Any]) -> bool:
        """Single attempt to run a test case. Returns True if successful, False if failed."""
        model_label = test_case['model']
        test_case_id = test_case['id']

        # Get model path
        model_path = self.model_paths.get(model_label)
        hf_model_path = self.hf_model_paths.get(model_label)

        # Use local path if it exists, otherwise use HF model path
        if model_path and os.path.exists(model_path):
            MODEL = model_path
            print(f"Using local model path: {MODEL}")
        else:
            if hf_model_path:
                MODEL = f"--model {hf_model_path}"
                print(f"Local path not found, using HF model: {hf_model_path}")
            else:
                print(f"Error: Neither local path nor HF model path found for {model_label}")
                return False

        # Generate extra-llm-api-config.yml
        config_content = self.generate_extra_llm_api_config(test_case)
        config_path = "/tmp/extra-llm-api-config.yml"

        with open(config_path, 'w') as f:
            f.write(config_content)

        print("extra-llm-api-config.yml:")
        print(config_content)

        # Build trtllm-serve command
        serve_cmd = [
            "trtllm-serve", MODEL, "--backend", "pytorch", "--tp_size",
            str(test_case['tp']), "--ep_size",
            str(test_case['ep']), "--max_batch_size",
            str(test_case['max_batch_size']), "--max_num_tokens",
            str(test_case['max_num_tokens']),
            "--kv_cache_free_gpu_memory_fraction",
            str(test_case['free_gpu_mem_fraction']), "--extra_llm_api_options",
            config_path
        ]

        print("Starting trtllm-serve with command:")
        print(' '.join(serve_cmd))
        print()

        # Start server
        server_log_filename = (
            f"trtllm-serve.{model_label}.tp{test_case['tp']}.ep{test_case['ep']}."
            f"attn{test_case['attn_backend']}.moe{test_case['moe_backend']}."
            f"gpu{test_case['free_gpu_mem_fraction']}.batch{test_case['max_batch_size']}."
            f"isl{test_case['isl']}.osl{test_case['osl']}."
            f"tokens{test_case['max_num_tokens']}.moetokens{test_case['moe_max_num_tokens']}.log"
        )

        server_process = None
        try:
            with open(server_log_filename, 'w') as log_file:
                log_file.write(f"extra-llm-api-config.yml:\n")
                log_file.write(config_content)
                log_file.write("\n")

            with open(server_log_filename, 'a') as log_file:
                server_process = subprocess.Popen(serve_cmd,
                                                  stdout=log_file,
                                                  stderr=subprocess.STDOUT)

            # Wait for server to be ready
            if not self.wait_for_server(server_process.pid,
                                        server_log_filename):
                print(
                    "Failed to start server, killing process and marking test case as failed"
                )
                return False

            # Run benchmarks based on execution plan
            for concurrency_index in self.execution_plan[test_case_id]:
                concurrency, iteration = test_case['concurrency_iterations'][concurrency_index]
                should_continue = self.run_benchmark(test_case, concurrency,
                                                     iteration, MODEL,
                                                     server_log_filename)

                # If run_benchmark returns False, mark test case as failed
                if not should_continue:
                    print(
                        f"RuntimeError detected - marking test case {test_case_id} as failed"
                    )
                    return False

            # If we reach here, all benchmarks completed successfully
            return True

        except Exception as e:
            print(f"Exception during test case execution: {e}")
            return False

        finally:
            # Cleanup: Kill server process using shell commands like in the original bash script
            if server_process:
                print(f"Stopping server for {model_label}")
                try:
                    # Use shell commands for more reliable process killing
                    subprocess.run(f"kill -9 {server_process.pid}",
                                   shell=True,
                                   check=False)
                    subprocess.run(f"wait {server_process.pid} 2>/dev/null || true",
                                   shell=True,
                                   check=False)
                except Exception as e:
                    print(f"Warning: Error during server cleanup: {e}")

                time.sleep(5)  # Give it time to clean up resources
                print(f"Server cleanup completed for {model_label}")

    def add_failed_test_case(self, test_case: Dict[str, Any], failure_reason: str) -> None:
        """Add a failed test case to the tracking list"""
        failed_info = {
            'test_case': test_case.copy(),
            'failure_reason': failure_reason,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'node_name': self.node_name,
            'gpu_info': self.gpu_info
        }
        self.failed_test_cases.append(failed_info)
        print(f"Added test case {test_case['id']} ({test_case['model']}) to failed test cases list")

    def generate_reproduction_script(self, failed_info: Dict[str, Any]) -> str:
        """Generate a shell script to reproduce a failed test case"""
        test_case = failed_info['test_case']
        test_case_id = test_case['id']
        model_label = test_case['model']
        
        # Get model path
        model_path = self.model_paths.get(model_label)
        hf_model_path = self.hf_model_paths.get(model_label)
        
        if not model_path:
            model_path = model_label
        else:
            # Use local path if it exists, otherwise use HF model path
            if not os.path.exists(model_path) and hf_model_path:
                model_path = hf_model_path
        
        script_content = f"""#!/bin/bash
# Reproduction script for failed test case {test_case_id} ({model_label})
# Generated on: {failed_info['timestamp']}
# Node: {failed_info['node_name']}
# Failure reason: {failed_info['failure_reason']}

set -e

echo "Reproducing failed test case {test_case_id} ({model_label})"
echo "Failure reason: {failed_info['failure_reason']}"
echo "Node: {failed_info['node_name']}"
echo "Timestamp: {failed_info['timestamp']}"
echo ""

# Set environment variables
export TQDM_MININTERVAL=1000
export PRINT_ITER_LOG=false

# Create output directory
OUTPUT_DIR="reproduction_output_{test_case_id}_{model_label}"
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "Created output directory: $OUTPUT_DIR"
echo ""

# Generate extra-llm-api-config.yml
cat > extra-llm-api-config.yml << 'EOF'
{self.generate_extra_llm_api_config(test_case)}
EOF

echo "Generated extra-llm-api-config.yml"
echo ""

# Determine model path (local or HF)
LOCAL_MODEL_PATH="{self.model_paths.get(model_label, '')}"
HF_MODEL_PATH="{self.hf_model_paths.get(model_label, '')}"

if [ -d "$LOCAL_MODEL_PATH" ]; then
    MODEL_PATH="$LOCAL_MODEL_PATH"
    echo "Using local model path: $MODEL_PATH"
else
    if [ -n "$HF_MODEL_PATH" ]; then
        MODEL_PATH="$HF_MODEL_PATH"
        echo "Local path not found, using HF model: $MODEL_PATH"
    else
        echo "Error: Neither local path nor HF model path found for {model_label}"
        exit 1
    fi
fi

# Build trtllm-serve command
SERVE_CMD=(
    "trtllm-serve" "$MODEL_PATH"
    "--backend" "pytorch"
    "--tp_size" "{test_case['tp']}"
    "--ep_size" "{test_case['ep']}"
    "--max_batch_size" "{test_case['max_batch_size']}"
    "--max_num_tokens" "{test_case['max_num_tokens']}"
    "--kv_cache_free_gpu_memory_fraction" "{test_case['free_gpu_mem_fraction']}"
    "--extra_llm_api_options" "extra-llm-api-config.yml"
)

echo "Starting trtllm-serve with command:"
echo "${{SERVE_CMD[@]}}"
echo ""

        # Generate server log filename
        SERVER_LOG="trtllm-serve.{model_label}.tp{test_case['tp']}.ep{test_case['ep']}.attn{test_case['attn_backend']}.moe{test_case['moe_backend']}.gpu{test_case['free_gpu_mem_fraction']}.batch{test_case['max_batch_size']}.isl{test_case['isl']}.osl{test_case['osl']}.tokens{test_case['max_num_tokens']}.moetokens{test_case['moe_max_num_tokens']}.log"

        # Start server in background
        "${{SERVE_CMD[@]}}" > "$SERVER_LOG" 2>&1 &
        SERVER_PID=$!

echo "Server started with PID: $SERVER_PID"
echo "Waiting for server to be ready..."

# Wait for server to be ready (simplified version)
for i in {{1..36}}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    if [ $i -eq 36 ]; then
        echo "Error: Server did not become ready after 6 minutes"
        kill -9 $SERVER_PID 2>/dev/null || true
        exit 1
    fi
    echo "Waiting for server... (attempt $i/36)"
    sleep 10
done

echo ""

# Run benchmarks for each concurrency level
"""
        
        # Add benchmark commands for each concurrency
        for concurrency_index in self.execution_plan.get(test_case_id, []):
            concurrency, iteration = test_case['concurrency_iterations'][concurrency_index]
            num_prompts = concurrency * iteration
            
            script_content += f"""
echo "Running benchmark for concurrency {concurrency_index + 1}: concurrency={concurrency}, iteration={iteration}, num-prompts={num_prompts}"
echo ""

BENCHMARK_CMD=(
    "python" "-m" "tensorrt_llm.serve.scripts.benchmark_serving"
    "--model" "$MODEL_PATH"
    "--dataset-name" "random"
    "--random-ids"
    "--num-prompts" "{num_prompts}"
    "--random-input-len" "{test_case['isl']}"
    "--random-output-len" "{test_case['osl']}"
    "--random-range-ratio" "0.0"
    "--ignore-eos"
    "--percentile-metrics" "ttft,tpot,itl,e2el"
    "--max-concurrency" "{concurrency}"
)

echo "Benchmark command:"
echo "${{BENCHMARK_CMD[@]}}"
echo ""

        # Generate benchmark log filename
        BENCHMARK_LOG="serve.{model_label}.tp{test_case['tp']}.ep{test_case['ep']}.attn{test_case['attn_backend']}.moe{test_case['moe_backend']}.gpu{test_case['free_gpu_mem_fraction']}.batch{test_case['max_batch_size']}.isl{test_case['isl']}.osl{test_case['osl']}.tokens{test_case['max_num_tokens']}.moetokens{test_case['moe_max_num_tokens']}.concurrency{concurrency}.iter{iteration}.log"

        # Run benchmark
        "${{BENCHMARK_CMD[@]}}" > "$BENCHMARK_LOG" 2>&1

if [ $? -eq 0 ]; then
    echo "Benchmark for concurrency {concurrency_index + 1} completed successfully"
else
    echo "Benchmark for concurrency {concurrency_index + 1} failed"
fi
echo "----------------------------------------"
"""
        
        script_content += f"""
echo ""
echo "All benchmarks completed for test case {test_case_id}"
echo ""

# Stop server
echo "Stopping server..."
kill -9 $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo "Server stopped"
echo "Reproduction script completed"
echo ""
        echo "Log files generated:"
        echo "- $SERVER_LOG: Server logs"
"""
        
        return script_content

    def generate_all_reproduction_scripts(self) -> None:
        """Generate reproduction scripts for all failed test cases"""
        if not self.failed_test_cases:
            print("No failed test cases to generate reproduction scripts for")
            return
        
        print(f"\nGenerating reproduction scripts for {len(self.failed_test_cases)} failed test cases...")
        
        generated_scripts = []
        for i, failed_info in enumerate(self.failed_test_cases, 1):
            try:
                test_case = failed_info['test_case']
                test_case_id = test_case['id']
                model_label = test_case['model']
                
                script_content = self.generate_reproduction_script(failed_info)
                script_filename = f"reproduce_test_case_{test_case_id}_{model_label}.sh"
                
                with open(script_filename, 'w') as f:
                    f.write(script_content)
                
                # Make script executable
                os.chmod(script_filename, 0o755)
                
                generated_scripts.append(script_filename)
                print(f"Generated reproduction script: {script_filename}")
                
            except Exception as e:
                print(f"Warning: Failed to generate reproduction script for test case {test_case_id}: {e}")
                continue
        
        if generated_scripts:
            print(f"\nSuccessfully generated {len(generated_scripts)} reproduction scripts in: {os.getcwd()}")
            print("You can run these scripts to reproduce the failed test cases:")
            for script in generated_scripts:
                print(f"  ./{script}")
        else:
            print("\nNo reproduction scripts were generated successfully")

    def run_benchmarks(self) -> None:
        """Main function to run all benchmarks from config file"""
        script_start_time = time.time()

        print(f"Using config file: {self.config_file}")
        if self.select_pattern:
            print(f"Select pattern: {self.select_pattern}")
        else:
            print("Select pattern: default (all test cases)")
        if self.skip_pattern:
            print(f"Skip pattern: {self.skip_pattern}")
        else:
            print("Skip pattern: default (no skipping)")

        # Load configuration
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)

        test_cases = config['test_cases']

        # Build execution plan
        self.build_execution_plan(test_cases)

        # Print execution plan before starting benchmarks
        self.print_execution_plan(test_cases)

        # Run each test case based on execution plan
        for i, test_case in enumerate(test_cases, 1):
            test_case_id = test_case['id']

            if test_case_id not in self.execution_plan:
                print("=" * 57)
                print(
                    f"Test case {i}/{len(test_cases)} (ID: {test_case_id}): {test_case['model']} - SKIPPED"
                )
                print("=" * 57)
                continue

            print("=" * 57)
            print(
                f"Test case {i}/{len(test_cases)} (ID: {test_case_id}): {test_case['model']}"
            )
            print(
                f"Config: GPUs={test_case['gpus']}, TP={test_case['tp']}, EP={test_case['ep']}, attn_backend={test_case['attn_backend']}, moe_backend={test_case['moe_backend']}"
            )
            print("=" * 57)

            self.run_test_case(test_case)

        # Calculate and display total script runtime
        script_total_time = time.time() - script_start_time
        hours = int(script_total_time // 3600)
        minutes = int((script_total_time % 3600) // 60)
        seconds = int(script_total_time % 60)

        print("=" * 80)
        print("SCRIPT COMPLETION SUMMARY")
        print("=" * 80)
        print(
            f"Total script runtime: {hours:02d}:{minutes:02d}:{seconds:02d} (HH:MM:SS)"
        )
        print(f"Total runtime in seconds: {script_total_time:.2f}")
        print("=" * 80)
        print("All benchmarks completed!")
        
        # Print summary of failed test cases
        if self.failed_test_cases:
            print("\n" + "=" * 80)
            print("FAILED TEST CASES SUMMARY")
            print("=" * 80)
            for i, failed_info in enumerate(self.failed_test_cases, 1):
                test_case = failed_info['test_case']
                print(f"{i}. Test Case {test_case['id']} ({test_case['model']})")
                print(f"   Failure Reason: {failed_info['failure_reason']}")
                print(f"   Timestamp: {failed_info['timestamp']}")
                print(f"   Node: {failed_info['node_name']}")
                print()
            print(f"Total failed test cases: {len(self.failed_test_cases)}")
            print("=" * 80)
        else:
            print("\nNo test cases failed during execution!")
        
        # Generate reproduction scripts for failed test cases
        self.generate_all_reproduction_scripts()


def main():
    parser = argparse.ArgumentParser(
        description='Run benchmarks from YAML configuration file')
    parser.add_argument('--output_folder',
                        required=True,
                        help='Output folder for benchmark results')
    parser.add_argument('--config_file',
                        required=True,
                        help='Path to YAML configuration file')
    parser.add_argument(
        '--skip',
        help=
        'Skip pattern: "2,4-1" means skip test case 2 and test case 4\'s 1st concurrency'
    )
    parser.add_argument(
        '--select',
        help=
        'Select pattern: "1,3,5" means only run test cases 1, 3, and 5; "1-1,2-3" means only run test case 1\'s 1st concurrency and test case 2\'s 3rd concurrency'
    )

    args = parser.parse_args()

    try:
        subprocess.run(f'echo "TRT-LLM GIT COMMIT": $TRT_LLM_GIT_COMMIT',
                       shell=True,
                       check=True)
    except subprocess.CalledProcessError:
        print("Warning: Could not echo TRT-LLM GIT COMMIT")

    if not os.path.exists(args.config_file):
        print(f"Error: Config file '{args.config_file}' does not exist")
        sys.exit(1)

    if not os.path.exists(args.output_folder):
        print(f"Error: Output folder '{args.output_folder}' does not exist")
        sys.exit(1)

    try:
        runner = BenchmarkRunner(args.output_folder, args.config_file,
                                 args.skip, args.select)
        runner.run_benchmarks()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
