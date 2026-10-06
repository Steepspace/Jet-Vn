import re
import sys
import shutil
import datetime
import textwrap
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor

from condor_utils.core.logging import setup_logging
from condor_utils.core.helpers import run_command_and_log, get_line_count, get_best_submit_node

class CondorJobManager:
    def __init__(self, args, job_name="Job"):
        self.args = args
        self.job_name = job_name
        self.output_dir = Path(args.output_dir).resolve()
        self.log_file = self.output_dir / 'log.txt'

        # Create output dir early so we can log
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging(self.log_file, logging.DEBUG)

        self.job_output_dir = Path(args.job_output_dir).resolve() if hasattr(args, 'job_output_dir') and args.job_output_dir else None

        # Commonly resolved paths
        self.input_list = Path(args.input_list).resolve() if hasattr(args, 'input_list') and args.input_list else None
        self.condor_script = Path(args.condor_script).resolve() if hasattr(args, 'condor_script') and args.condor_script else None
        self.condor_log_dir = Path(args.condor_log_dir).resolve() if hasattr(args, 'condor_log_dir') and args.condor_log_dir else None
        self.common_errors = Path(args.common_errors).resolve() if hasattr(args, 'common_errors') and args.common_errors else None

        self.files_to_check = []
        self.dirs_to_check = []
        if self.input_list:
            self.files_to_check.append(self.input_list)
        if self.condor_script:
            self.files_to_check.append(self.condor_script)
        if self.common_errors:
            self.files_to_check.append(self.common_errors)

    def add_file_to_check(self, path):
        if path:
            self.files_to_check.append(Path(path).resolve())

    def add_dir_to_check(self, path):
        if path:
            self.dirs_to_check.append(Path(path).resolve())

    def validate_paths(self):
        for f in self.files_to_check:
            if not f.is_file():
                self.logger.critical(f'File: {f} does not exist!')
                sys.exit(1)
        for d in self.dirs_to_check:
            if not d.is_dir():
                self.logger.critical(f'Directory: {d} does not exist!')
                sys.exit(1)

    def get_best_submit_node(self):
        if not hasattr(self, '_ranked_nodes') or self._ranked_nodes is None:
            manual_node = getattr(self.args, 'node', None)
            user = getattr(self.args, 'user', None) or "anarde"
            detected_nodes, node_status = get_best_submit_node(logger=self.logger, user=user)
            self._ranked_nodes = [manual_node] if manual_node else detected_nodes
            self._node_status = node_status
        return self._ranked_nodes, self._node_status

    def get_monitor_command(self, interval="60s", background=True):
        project_root = Path(__file__).resolve().parents[2]
        monitor_script = (project_root / "scripts" / "condor" / "monitor_jobs.py").resolve()
        email = getattr(self.args, 'email', None) or "<email>"
        if background:
            log_file = self.output_dir / "monitor.log"
            return f"nohup python3 -u {monitor_script} -d {self.output_dir} -i {interval} -e {email} > {log_file} 2>&1 &"
        return f"python3 {monitor_script} -d {self.output_dir} -i {interval} -e {email}"

    def log_initialization(self, extra_logs=None):
        total_files = get_line_count(self.input_list) if self.input_list else 0
        self.logger.info('#'*40)
        self.logger.info(f'LOGGING: {datetime.datetime.now()}')
        self.logger.info(f'Job Name: {self.job_name}')
        if self.input_list:
            self.logger.info(f'Input DST List: {self.input_list}')
        self.logger.info(f'Total Runs: {total_files}')
        if hasattr(self.args, 'dst_per_job'):
            self.logger.info(f'DST Per Job: {self.args.dst_per_job}')
        if hasattr(self.args, 'events'):
            self.logger.info(f'Events to process: {self.args.events if self.args.events != 0 else "All"}')
        if hasattr(self.args, 'dbtag'):
            self.logger.info(f'DB Tag: {self.args.dbtag}')
        self.logger.info(f'Output Directory: {self.output_dir}')
        if self.job_output_dir:
            self.logger.info(f'Job Output Directory: {self.job_output_dir}')
        self.logger.info(f'Log File: {self.log_file}')
        if hasattr(self.args, 'memory'):
            self.logger.info(f'Condor Memory: {self.args.memory} GB')
            retry_mem = getattr(self.args, 'retry_request_memory', None)
            step = getattr(self.args, 'retry_memory_step', 0.5)
            ceiling = getattr(self.args, 'retry_memory_max', 6.0)
            if retry_mem:
                self.logger.info(f'Retry Request Memory: {retry_mem}')
            elif step and step > 0 and self.args.memory is not None:
                self.logger.info(f'Retry Memory Policy: +{step} GB on OOM eviction up to ceiling {ceiling} GB')
        if self.condor_script:
            self.logger.info(f'Condor Script: {self.condor_script}')
        if self.condor_log_dir:
            self.logger.info(f'Condor Log Directory: {self.condor_log_dir}')
        if self.common_errors:
            self.logger.info(f'Common Errors File: {self.common_errors}')
        if self.job_name != "hadd":
            ranked_nodes, node_status = self.get_best_submit_node()
            top_node = ranked_nodes[0]
            if node_status:
                user = getattr(self.args, 'user', None) or "anarde"
                status_summary = ", ".join(
                    f"{n}: {node_status[n]['user_total']} ({node_status[n]['user_running']}R/{node_status[n]['user_idle']}I)"
                    for n in sorted(node_status.keys())
                )
                self.logger.info(f'Submit Nodes User Jobs ({user}): {status_summary}')
                self.logger.info(
                    f'Top Submit Node: {top_node} '
                    f'({node_status[top_node]["user_total"]} jobs for {user} '
                    f'[{node_status[top_node]["user_running"]} running, {node_status[top_node]["user_idle"]} idle] | '
                    f'node total: {node_status[top_node]["total_running"]} running)'
                )
            else:
                self.logger.info(f'Top Submit Node: {top_node}')
            self.logger.info(f'Monitor Command (nohup): {self.get_monitor_command(background=True)}')

        if extra_logs:
            for k, v in extra_logs.items():
                self.logger.info(f'{k}: {v}')

        return total_files

    def prepare_directories(self):
        if self.condor_log_dir:
            shutil.rmtree(self.condor_log_dir, ignore_errors=True)
            self.condor_log_dir.mkdir(parents=True, exist_ok=True)

        for subdir in ['stdout', 'error']:
            shutil.rmtree(self.output_dir / subdir, ignore_errors=True)
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

        output_symlink = self.output_dir / 'output'
        if self.job_output_dir:
            self.job_output_dir.mkdir(parents=True, exist_ok=True)
            if output_symlink.is_symlink() or output_symlink.is_file():
                output_symlink.unlink()
            elif output_symlink.is_dir():
                shutil.rmtree(output_symlink)
            output_symlink.symlink_to(self.job_output_dir, target_is_directory=True)
        else:
            shutil.rmtree(output_symlink, ignore_errors=True)
            output_symlink.mkdir(parents=True, exist_ok=True)

        files_dir = self.output_dir / 'files'
        files_dir.mkdir(parents=True, exist_ok=True)
        return files_dir

    def copy_dependencies(self, extra_files=None, extra_dirs=None):
        if self.input_list:
            shutil.copy(self.input_list, self.output_dir)
        if self.condor_script:
            shutil.copy(self.condor_script, self.output_dir)
        if self.common_errors:
            shutil.copy(self.common_errors, self.output_dir)

        if extra_files:
            for f in extra_files:
                if f:
                    shutil.copy(Path(f).resolve(), self.output_dir)

        if extra_dirs:
            for d in extra_dirs:
                if d:
                    src = Path(d).resolve()
                    shutil.copytree(src, self.output_dir / src.name, dirs_exist_ok=True)

    def prepare_job_lists(self, dst_per_job, files_dir=None, jobs_file_name="jobs.list", max_workers=16, calib_map=None):
        """
        Splits input lists into chunks of dst_per_job and writes resolved chunk paths to jobs_file_name.
        Uses multithreading and native Python file I/O for speed across thousands of runs.
        If calib_map is provided, appends the corresponding calibration file (or 'default') to each entry.
        """
        if files_dir is None:
            files_dir = self.output_dir / 'files'
        files_dir = Path(files_dir).resolve()
        files_dir.mkdir(parents=True, exist_ok=True)

        jobs_file = self.output_dir / jobs_file_name
        jobs_file.unlink(missing_ok=True)

        if not self.input_list or not self.input_list.is_file():
            self.logger.warning("Input list is not set or does not exist.")
            return []

        input_files = [line.strip() for line in self.input_list.read_text(encoding='utf-8').splitlines() if line.strip()]
        total_runs = len(input_files)
        self.logger.info(f"Preparing job lists for {total_runs} runs (dst_per_job={dst_per_job})...")

        if dst_per_job <= 0:
            dst_per_job = 1

        def process_run(run_file_str):
            run_path = Path(run_file_str)
            if not run_path.is_file():
                candidate = self.input_list.parent / run_file_str
                if candidate.is_file():
                    run_path = candidate
                else:
                    self.logger.warning(f"Run list file not found: {run_file_str}")
                    return []
            stem = run_path.stem
            try:
                lines = [l.strip() for l in run_path.read_text(encoding='utf-8').splitlines() if l.strip()]
            except Exception as e:
                self.logger.error(f"Failed to read {run_path}: {e}")
                return []

            if not lines:
                return []

            calib_suffix = ""
            if calib_map is not None:
                parts = stem.split('-')
                if len(parts) >= 2 and any(c.isdigit() for c in parts[1]):
                    run = parts[1].lstrip('0')
                else:
                    match = re.search(r'\b(\d+)\b', stem)
                    run = match.group(1).lstrip('0') if match else stem.lstrip('0')
                calib_suffix = f",{calib_map.get(run, 'default')}"

            job_paths = []
            for idx, chunk_start in enumerate(range(0, len(lines), dst_per_job)):
                chunk = lines[chunk_start:chunk_start + dst_per_job]
                chunk_file = files_dir / f"{stem}-{idx:03d}.list"
                chunk_file.write_text("\n".join(chunk) + "\n", encoding='utf-8')
                job_paths.append(f"{chunk_file.resolve()}{calib_suffix}")
            return job_paths

        all_job_paths = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for run_jobs in executor.map(process_run, input_files):
                all_job_paths.extend(run_jobs)

        if all_job_paths:
            jobs_file.write_text("\n".join(all_job_paths) + "\n", encoding='utf-8')

        self.logger.info(f"Total jobs created: {len(all_job_paths)} across {total_runs} runs.")
        return all_job_paths

    def write_submit_file(
        self,
        arguments,
        executable=None,
        memory=None,
        retry_request_memory=None,
        retry_memory_step=None,
        retry_memory_max=None,
        max_retries=None,
        sub_file_name="genFun4All.sub",
        stdout_dir="stdout",
        error_dir="error",
        log_prefix="job",
    ):
        exec_file = executable or (self.condor_script.name if self.condor_script else "script.sh")
        mem = memory if memory is not None else getattr(self.args, 'memory', 1.0)

        # Parse base memory
        mem_float = None
        if isinstance(mem, (int, float)):
            mem_float = float(mem)
            mem_str = f"{int(mem_float)}GB" if mem_float.is_integer() else f"{mem_float:g}GB"
        else:
            mem_s = str(mem).strip()
            m = re.match(r"^([\d\.]+)\s*([a-zA-Z]*)$", mem_s)
            if m:
                mem_float = float(m.group(1))
                unit = m.group(2) or "GB"
                mem_str = f"{mem_float:g}{unit}"
            else:
                mem_str = mem_s

        # Determine retry_request_memory
        retry_mem = retry_request_memory
        if retry_mem is None:
            retry_mem = getattr(self.args, 'retry_request_memory', None)

        if retry_mem is None:
            step = retry_memory_step if retry_memory_step is not None else getattr(self.args, 'retry_memory_step', 0.5)
            ceiling = retry_memory_max if retry_memory_max is not None else getattr(self.args, 'retry_memory_max', 6.0)
            if step and step > 0 and mem_float is not None and ceiling and ceiling > mem_float:
                steps = []
                curr = mem_float + float(step)
                while round(curr, 4) <= round(float(ceiling), 4):
                    val = round(curr, 4)
                    steps.append(f"{int(val)}GB" if val.is_integer() else f"{val:g}GB")
                    curr += float(step)
                if steps:
                    retry_mem = ", ".join(steps)

        log_dir = self.condor_log_dir or (self.output_dir / 'logs')

        # Ensure max_retries has enough attempts for all retry_request_memory tiers
        num_memory_tiers = len([s for s in retry_mem.split(',') if s.strip()]) if retry_mem else 0
        base_retries = max_retries if max_retries is not None else getattr(self.args, 'max_retries', 3)
        effective_retries = max(int(base_retries), num_memory_tiers)
        if num_memory_tiers > int(base_retries):
            self.logger.info(
                f"Elevating max_retries from {base_retries} to {effective_retries} "
                f"to cover all {num_memory_tiers} memory retry tiers."
            )

        lines = [
            f"executable           = {exec_file}",
            f"arguments            = {arguments}",
            f"log                  = {log_dir}/{log_prefix}-$(ClusterId)-$(Process).log",
            f"output               = {stdout_dir}/job-$(ClusterId)-$(Process).out",
            f"error                = {error_dir}/job-$(ClusterId)-$(Process).err",
            f"request_memory       = {mem_str}",
        ]
        if retry_mem:
            lines.append(f"retry_request_memory = {retry_mem}")
        lines.extend([
            f"max_retries          = {effective_retries}",
            f"stream_output        = True",
            f"stream_error         = True",
            "",
        ])

        sub_file = self.output_dir / sub_file_name
        sub_file.write_text("\n".join(lines))
        return sub_file

    def finalize_submission(self, queue_arg="input_dst from jobs.list", sub_file_name="genFun4All.sub", limit=15000, execute=False, use_ssh=None, clean_log_dir=None):
        match = re.search(r" from ([\w\.-]+)", queue_arg)
        list_file = match.group(1) if match else "jobs.list"
        list_path = self.output_dir / list_file

        total_lines = get_line_count(list_path)

        if total_lines > limit:
            # Split the job list file to respect the Condor submission limit using native Python
            lines = [l for l in list_path.read_text(encoding='utf-8').splitlines() if l.strip()]
            split_prefix = "jobs-"
            split_files = []
            for i, chunk_start in enumerate(range(0, len(lines), limit)):
                chunk = lines[chunk_start:chunk_start + limit]
                split_file = self.output_dir / f"{split_prefix}{i}.list"
                split_file.write_text("\n".join(chunk) + "\n", encoding='utf-8')
                split_files.append(split_file)
            if not split_files:
                split_files = [list_path]
        else:
            split_files = [list_path]

        if use_ssh is None:
            use_ssh = not execute
        if clean_log_dir is None:
            clean_log_dir = not execute

        ranked_nodes = None
        if use_ssh:
            ranked_nodes, _ = self.get_best_submit_node()

        log_dir = self.condor_log_dir or (self.output_dir / 'logs')
        prep_cmd = f"rm -rf {log_dir} && mkdir -p {log_dir} && " if clean_log_dir else ""

        for i, sf in enumerate(split_files):
            current_queue_arg = queue_arg.replace(list_file, sf.name)
            base_cmd = f'{prep_cmd}cd {self.output_dir} && condor_submit {sub_file_name} -queue "{current_queue_arg}"'
            if use_ssh:
                target_node = ranked_nodes[i % len(ranked_nodes)]
                command = f"ssh {target_node} '{base_cmd}'"
            else:
                command = base_cmd

            if execute:
                run_command_and_log(command, self.logger, self.output_dir)
            else:
                self.logger.info(command)

        if not execute:
            monitor_bg_cmd = self.get_monitor_command(background=True)
            log_file = self.output_dir / "monitor.log"
            self.logger.info(f'To monitor job status in the background (recommended):\n  {monitor_bg_cmd}')
            self.logger.info(f'To view monitor log live:\n  tail -f {log_file}')
