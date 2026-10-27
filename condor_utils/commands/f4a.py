import math
import re
import shutil
import sys
from pathlib import Path
from condor_utils.core.manager import CondorJobManager
from condor_utils.core.helpers import chunk_list
from condor_utils.cli import get_common_parser, add_build_args

def parse_calib_list(calib_list, logger=None):
    """
    Parses a calibration list file and returns a mapping from run number (str) to calibration line/path.
    Extracts run number from directory or filename, normalizes by stripping leading zeros,
    and ignores empty or commented lines.
    """
    calib_map = {}
    if not calib_list:
        return calib_map

    calib_path = Path(calib_list)
    if not calib_path.is_file():
        if logger:
            logger.warning(f"Calib list file does not exist: {calib_list}")
        return calib_map

    for line in calib_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        raw_path = line.split(',')[0].strip()
        path_obj = Path(raw_path)

        # Prefer parent directory if numeric (e.g. /CDB/<run>/<file>); fallback to filename pattern (e.g. -<run>v2.root or -<run>.root)
        if len(path_obj.parts) >= 2 and path_obj.parts[-2].isdigit():
            run = path_obj.parts[-2].lstrip('0')
        else:
            match = re.search(r'-(\d+)(?:v\d+)?\.root$', path_obj.name)
            run = match.group(1).lstrip('0') if match else None

        if run:
            if logger:
                logger.debug(f'Processing Calib: {line}, run: {run}')
            calib_map[run] = line
        else:
            if logger:
                logger.warning(f'Could not parse run number from calib entry: {line}')

    if logger:
        logger.info(f'Loaded {len(calib_map)} calibration entries from {calib_path.name}')

    return calib_map

ZS_DETECTORS = ('CEMC', 'HCALIN', 'HCALOUT')

def parse_zs_calib_list(zs_list, logger=None):
    """
    Parses a ZS cross calib T0 list file (<DET>_ZSCrossCalibT0_..._<run>.root, one file per line)
    and returns a mapping from run number (str, leading zeros stripped) to {detector: path}.
    """
    zs_map = {}
    if not zs_list:
        return zs_map

    zs_path = Path(zs_list)
    if not zs_path.is_file():
        if logger:
            logger.warning(f"ZS calib list file does not exist: {zs_list}")
        return zs_map

    pattern = re.compile(rf"^({'|'.join(ZS_DETECTORS)})_ZSCrossCalibT0_.*_(\d+)\.root$")
    for line in zs_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        match = pattern.match(Path(line).name)
        if not match:
            if logger:
                logger.warning(f'Could not parse detector/run from ZS calib entry: {line}')
            continue

        det, run = match.group(1), match.group(2).lstrip('0')
        zs_map.setdefault(run, {})[det] = line

    for run, entry in sorted(zs_map.items(), key=lambda kv: int(kv[0])):
        missing = [det for det in ZS_DETECTORS if det not in entry]
        if missing and logger:
            logger.warning(f"Run {run} ZS calib list is missing {', '.join(missing)}; passing none for those")

    if logger:
        logger.info(f'Loaded ZS cross calib T0 entries for {len(zs_map)} runs from {zs_path.name}')

    return zs_map

EVENTS_PER_SEGMENT = 100000

def parse_run_event_list(event_list, logger=None):
    """
    Parses a run,event event list (header 'run,event', e.g. from scripts/data/high_pt_event_dsts.py)
    into a mapping (run, segment) -> sorted event IDs, with segment = event // EVENTS_PER_SEGMENT.
    Run and segment are strings with leading zeros stripped. Returns None for a plain event list
    (one event ID per line), which is passed unchanged to every job.
    """
    lines = [l.strip() for l in Path(event_list).read_text(encoding='utf-8').splitlines() if l.strip() and not l.startswith('#')]
    if not lines or lines[0].replace(' ', '').lower() != 'run,event':
        return None

    run_events = {}
    for line in lines[1:]:
        cols = [c.strip() for c in line.split(',')]
        try:
            run, event = int(cols[0]), int(cols[1])
        except (IndexError, ValueError):
            if logger:
                logger.warning(f'Skipping event list line without run,event: {line}')
            continue
        run_events.setdefault((str(run), str(event // EVENTS_PER_SEGMENT)), set()).add(event)

    if logger:
        n_events = sum(len(v) for v in run_events.values())
        logger.info(f'Loaded {n_events} events in {len(run_events)} segments from {Path(event_list).name}')

    return {key: sorted(events) for key, events in run_events.items()}

def write_job_event_lists(manager, run_events, jobs_file_name="jobs.list"):
    """
    Gives each job of jobs_file_name its own event list with the events of its segments (one event ID
    per line, next to its DST chunk) as an extra column. Jobs without any listed event are dropped,
    since an empty event list would make EventSkip process every event.
    """
    jobs_file = manager.output_dir / jobs_file_name
    kept = []
    covered = set()
    n_dropped = 0
    for job in [l for l in jobs_file.read_text(encoding='utf-8').splitlines() if l.strip()]:
        chunk = Path(job.split(',')[0])
        events = []
        for line in chunk.read_text(encoding='utf-8').splitlines():
            match = re.search(r'-(\d+)-(\d+)\.root$', Path(line.split(',')[0].strip()).name)
            if not match:
                continue
            key = (match.group(1).lstrip('0'), str(int(match.group(2))))
            if key in run_events:
                covered.add(key)
                events.extend(run_events[key])
        if not events:
            n_dropped += 1
            continue
        events_file = chunk.with_suffix('.events')
        events_file.write_text("\n".join(map(str, sorted(events))) + "\n", encoding='utf-8')
        kept.append(f"{job},{events_file}")

    if not kept:
        manager.logger.critical("No job has any event of the run,event list: check that the input DSTs cover its runs and segments.")
        sys.exit(1)
    jobs_file.write_text("\n".join(kept) + "\n", encoding='utf-8')

    missing = sorted(set(run_events) - covered, key=lambda k: (int(k[0]), int(k[1])))
    if missing:
        n_missing = sum(len(run_events[k]) for k in missing)
        segs_str = ", ".join(f"{run}-{seg}" for run, seg in missing)
        manager.logger.warning(f"{n_missing} listed event(s) in {len(missing)} segment(s) not covered by any input DST: {segs_str}")
    manager.logger.info(f"Event list split per job: kept {len(kept)} job(s), dropped {n_dropped} without listed events.")

def create_f4a_jobs(args):
    if getattr(args, 'overlay', False):
        return create_f4a_overlay_jobs(args)

    if "Fun4All_BkgSub" in args.f4a_macro:
        args.condor_script = 'scripts/genFun4All_BkgSub.sh'

    manager = CondorJobManager(args, job_name="F4A")
    manager.add_file_to_check(args.f4a_macro)
    manager.add_file_to_check(args.calo_calib_macro)
    manager.add_file_to_check(args.HIJetReco_macro)
    manager.add_dir_to_check(args.src_dir)

    # the eta calib is only used by the mult subtraction, so it implies --do-mult 1
    # unless --do-mult was given explicitly
    if getattr(args, 'do_mult', 0) is None:
        args.do_mult = 1 if getattr(args, 'eta_calib_path', None) else 0
        if args.do_mult:
            manager.logger.info('--eta-calib-path given without --do-mult: enabling mult subtraction')

    calib_list = Path(args.calib).resolve() if args.calib else None
    if calib_list:
        manager.add_file_to_check(calib_list)

    eta_calib = Path(args.eta_calib_path).resolve() if getattr(args, 'eta_calib_path', None) else None
    if eta_calib:
        manager.add_file_to_check(eta_calib)

    event_list = Path(args.event_list).resolve() if getattr(args, 'event_list', None) else None
    if event_list:
        manager.add_file_to_check(event_list)

    zs_calib_list = Path(args.zs_calib).resolve() if getattr(args, 'zs_calib', None) else None
    if zs_calib_list:
        manager.add_file_to_check(zs_calib_list)

    manager.validate_paths()

    init_log = {
        'Calib List': calib_list if calib_list else "Not Provided (Using default)",
        'Eta Calib': eta_calib if eta_calib else "Not Provided (Using default/empty)",
        'Event List': event_list if event_list else "Not Provided (Using all events)",
        'Fun4All Macro': Path(args.f4a_macro).resolve(),
        'Calo Calib Macro': Path(args.calo_calib_macro).resolve(),
        'HIJetReco Macro': Path(args.HIJetReco_macro).resolve(),
        'Source Directory': Path(args.src_dir).resolve()
    }
    if "Fun4All_BkgSub" in args.f4a_macro:
        init_log['Do RCone'] = bool(getattr(args, 'do_rcone', False))
        init_log['Do Mult'] = bool(getattr(args, 'do_mult', 0))
        init_log['Do Neg Energy Threshold'] = bool(getattr(args, 'do_neg_energy_threshold', 1))
        init_log['Neg Energy Threshold'] = float(getattr(args, 'neg_energy_threshold', -2.0))
        init_log['ZS Calib List'] = zs_calib_list if zs_calib_list else "Not Provided (Using CDB)"
        init_log['Do EMfrac Calib'] = bool(getattr(args, 'do_emfrac_calib', 0))

    manager.log_initialization(init_log)

    files_dir = manager.prepare_directories()

    extra_files = [args.f4a_macro, args.calo_calib_macro, args.HIJetReco_macro]
    if calib_list:
        extra_files.append(calib_list)
    if eta_calib:
        extra_files.append(eta_calib)
    if event_list:
        extra_files.append(event_list)
    if zs_calib_list:
        extra_files.append(zs_calib_list)

    manager.copy_dependencies(extra_files=extra_files, extra_dirs=[args.src_dir])

    calib_map = parse_calib_list(calib_list, manager.logger)

    # ZS cross calib T0 columns (BkgSub only): one file per detector, 'none' falls back to the CDB
    zs_columns = None
    missing_zs_runs = set()
    if "Fun4All_BkgSub" in args.f4a_macro:
        zs_map = parse_zs_calib_list(zs_calib_list, manager.logger)

        def zs_columns(run):
            entry = zs_map.get(run, {})
            if not entry:
                missing_zs_runs.add(run)
            return ",".join(entry.get(det, 'none') for det in ZS_DETECTORS)

    manager.prepare_job_lists(dst_per_job=args.dst_per_job, files_dir=files_dir, calib_map=calib_map, extra_columns=zs_columns)

    if zs_calib_list and missing_zs_runs:
        runs_str = ", ".join(sorted(missing_zs_runs, key=lambda r: int(r) if r.isdigit() else r))
        manager.logger.warning(f"No ZS cross calib T0 entry for {len(missing_zs_runs)} run(s), passing none (CDB fallback): {runs_str}")

    # run,event list (BkgSub only): each job gets the events of its own segments as a jobs.list column
    run_events = None
    if event_list and "Fun4All_BkgSub" in args.f4a_macro:
        run_events = parse_run_event_list(event_list, manager.logger)
        if run_events is not None:
            write_job_event_lists(manager, run_events)

    if "Fun4All_BkgSub" in args.f4a_macro:
        eta_calib_val = (manager.output_dir / eta_calib.name) if eta_calib else "none"
        if run_events is not None:
            event_list_val = "$(event_list)"
        else:
            event_list_val = (manager.output_dir / event_list.name) if event_list else "none"
        do_rcone_val = 1 if getattr(args, 'do_rcone', False) else 0
        do_mult_val = getattr(args, 'do_mult', 0)
        do_neg_energy_threshold_val = getattr(args, 'do_neg_energy_threshold', 1)
        neg_energy_threshold_val = getattr(args, 'neg_energy_threshold', -2.0)
        do_emfrac_calib_val = getattr(args, 'do_emfrac_calib', 0)
        bkgsub_args = f"{args.do_flow} {eta_calib_val} {event_list_val} {do_rcone_val} {do_mult_val} {do_neg_energy_threshold_val} {neg_energy_threshold_val} $(zs_cemc) $(zs_hcalin) $(zs_hcalout) {do_emfrac_calib_val} "
        queue_arg = "input_dst,input_calib,zs_cemc,zs_hcalin,zs_hcalout from jobs.list"
        if run_events is not None:
            queue_arg = "input_dst,input_calib,zs_cemc,zs_hcalin,zs_hcalout,event_list from jobs.list"
    else:
        bkgsub_args = ""
        queue_arg = "input_dst,input_calib from jobs.list"

    arguments = f"{manager.output_dir / Path(args.f4a_macro).name} $(input_dst) $(input_calib) test-$(ClusterId)-$(Process).root tree-$(ClusterId)-$(Process).root {args.events} {args.dbtag} {bkgsub_args}{manager.output_dir}/output"
    manager.write_submit_file(arguments=arguments)
    manager.finalize_submission(queue_arg=queue_arg)

def overlay_run_segment(line):
    """
    (run, segment) of an overlay list line (DST_GLOBAL,DST_TRUTH_JET,DST_TRUTH_G4HIT,DST_CALO), taken from
    the DST_CALO name (DST_CALO_Jet12-00000021-data-00068144-00000.root), leading zeros stripped.
    """
    parts = Path(line.split(',')[-1].strip()).stem.split('-')
    if len(parts) < 5:
        return None, None
    return parts[3].lstrip('0'), str(int(parts[4]))

def parse_data_zdc_sepd_list(data_list, logger=None):
    """
    Parses data lists (calofit,zdc,sepd per line, as used by Fun4All_BkgSub) into a mapping
    (run, segment) -> (zdc, sepd). data_list is one such list or a list of such lists.
    """
    data_map = {}
    lines = [l.strip() for l in Path(data_list).read_text(encoding='utf-8').splitlines() if l.strip() and not l.startswith('#')]
    if lines and ',' not in lines[0]:
        sub_lines = []
        for sub_list in lines:
            sub_lines.extend(l.strip() for l in Path(sub_list).read_text(encoding='utf-8').splitlines() if l.strip() and not l.startswith('#'))
        lines = sub_lines

    for line in lines:
        cols = [c.strip() for c in line.split(',')]
        if len(cols) != 3:
            if logger:
                logger.warning(f'Skipping data list line without 3 columns (calofit,zdc,sepd): {line}')
            continue
        match = re.search(r'-(\d+)-(\d+)\.root$', Path(cols[0]).name)
        if not match:
            if logger:
                logger.warning(f'Could not parse run/segment from data list entry: {line}')
            continue
        data_map[(match.group(1).lstrip('0'), str(int(match.group(2))))] = (cols[1], cols[2])

    if logger:
        logger.info(f'Loaded ZDC/sEPD DSTs for {len(data_map)} segments from {Path(data_list).name}')

    return data_map

def create_f4a_overlay_jobs(args):
    """
    Data+MC overlay mode of Fun4All_BkgSub: one job per overlay list line (one OutDir, all its events).
    The data-only DST_CALOFITTING next to the overlay DSTs is located by the condor script; the data
    ZDC/sEPD DSTs of the same run and segment come from the data list (--data-list).
    """
    if "Fun4All_BkgSub" not in args.f4a_macro:
        raise ValueError(f"--overlay is only supported by Fun4All_BkgSub, got {args.f4a_macro}")
    if not getattr(args, 'data_list', None):
        raise ValueError("--overlay needs --data-list (calofit,zdc,sepd lists of the data runs)")
    args.condor_script = 'scripts/genFun4All_BkgSub_Overlay.sh'

    manager = CondorJobManager(args, job_name="F4A Overlay")
    manager.add_file_to_check(args.f4a_macro)
    data_list = Path(args.data_list).resolve()
    manager.add_file_to_check(data_list)
    manager.add_file_to_check(args.calo_calib_macro)
    manager.add_file_to_check(args.HIJetReco_macro)
    manager.add_dir_to_check(args.src_dir)

    calib_list = Path(args.calib).resolve() if args.calib else None
    if calib_list:
        manager.add_file_to_check(calib_list)

    eta_calib = Path(args.eta_calib_path).resolve() if getattr(args, 'eta_calib_path', None) else None
    if eta_calib:
        manager.add_file_to_check(eta_calib)

    zs_calib_list = Path(args.zs_calib).resolve() if getattr(args, 'zs_calib', None) else None
    if zs_calib_list:
        manager.add_file_to_check(zs_calib_list)

    manager.validate_paths()

    manager.log_initialization({
        'Data List (ZDC/sEPD)': data_list,
        'Calib List': calib_list if calib_list else "Not Provided (Using default)",
        'Eta Calib': eta_calib if eta_calib else "Not Provided (Using default/empty)",
        'ZS Calib List': zs_calib_list if zs_calib_list else "Not Provided (Using CDB)",
        'Do RCone': bool(getattr(args, 'do_rcone', False)),
        'Do Neg Energy Threshold': bool(getattr(args, 'do_neg_energy_threshold', 1)),
        'Neg Energy Threshold': float(getattr(args, 'neg_energy_threshold', -2.0)),
        'Do EMfrac Calib': bool(getattr(args, 'do_emfrac_calib', 0)),
        'Jet pT Min': f"{args.jet_pt_min} GeV",
        'Truth Jet pT Min': f"{args.truth_jet_pt_min} GeV",
        'Fun4All Macro': Path(args.f4a_macro).resolve(),
        'Calo Calib Macro': Path(args.calo_calib_macro).resolve(),
        'HIJetReco Macro': Path(args.HIJetReco_macro).resolve(),
        'Source Directory': Path(args.src_dir).resolve()
    })

    manager.prepare_directories()

    extra_files = [args.f4a_macro, args.calo_calib_macro, args.HIJetReco_macro, data_list]
    if calib_list:
        extra_files.append(calib_list)
    if eta_calib:
        extra_files.append(eta_calib)
    if zs_calib_list:
        extra_files.append(zs_calib_list)

    manager.copy_dependencies(extra_files=extra_files, extra_dirs=[args.src_dir])

    calib_map = parse_calib_list(calib_list, manager.logger)
    zs_map = parse_zs_calib_list(zs_calib_list, manager.logger)
    data_map = parse_data_zdc_sepd_list(data_list, manager.logger)

    jobs_file = manager.output_dir / 'jobs.list'
    jobs_file.unlink(missing_ok=True)

    # -i is an overlay list, or a list of overlay lists (e.g. one per pT-hat sample)
    overlay_lines = [l.strip() for l in manager.input_list.read_text(encoding='utf-8').splitlines() if l.strip() and not l.startswith('#')]
    if overlay_lines and ',' not in overlay_lines[0]:
        sub_lines = []
        for sub_list in overlay_lines:
            sub_path = Path(sub_list)
            if not sub_path.is_file():
                manager.logger.warning(f"Overlay sub-list file not found: {sub_list}")
                continue
            lines = [l.strip() for l in sub_path.read_text(encoding='utf-8').splitlines() if l.strip() and not l.startswith('#')]
            manager.logger.info(f"Overlay list {sub_path.name}: {len(lines)} lines")
            sub_lines.extend(lines)
        overlay_lines = sub_lines

    total_jobs = 0
    missing_segments = set()
    with open(jobs_file, mode='w', encoding='utf-8') as file_out:
        for line in overlay_lines:

            if len(line.split(',')) != 4:
                manager.logger.warning(f"Skipping overlay line without 4 columns (global,truth_jet,g4hit,calo): {line}")
                continue

            run, segment = overlay_run_segment(line)
            if (run, segment) not in data_map:
                missing_segments.add(f'{run}-{segment}')
                continue

            zdc, sepd = data_map[(run, segment)]
            calib = calib_map.get(run, 'default')
            zs_entry = zs_map.get(run, {})
            zs_columns = ",".join(zs_entry.get(det, 'none') for det in ZS_DETECTORS)
            file_out.write(f'{line},{zdc},{sepd},{calib},{zs_columns}\n')
            total_jobs += 1

    if missing_segments:
        manager.logger.warning(f"Skipped overlay lines without data ZDC/sEPD DSTs for {len(missing_segments)} run-segment(s): {', '.join(sorted(missing_segments))}")
    manager.logger.info(f"Total jobs prepared: {total_jobs}")

    eta_calib_val = (manager.output_dir / eta_calib.name) if eta_calib else "none"
    do_rcone_val = 1 if getattr(args, 'do_rcone', False) else 0
    do_neg_energy_threshold_val = getattr(args, 'do_neg_energy_threshold', 1)
    neg_energy_threshold_val = getattr(args, 'neg_energy_threshold', -2.0)
    do_emfrac_calib_val = getattr(args, 'do_emfrac_calib', 0)

    arguments = (f"{manager.output_dir / Path(args.f4a_macro).name} $(input_dst_global) $(input_dst_jet) $(input_dst_g4hit) $(input_dst_calo) $(input_dst_zdc) $(input_dst_sepd) $(input_calib) "
                 f"test-$(ClusterId)-$(Process).root tree-$(ClusterId)-$(Process).root {args.events} {args.dbtag} {args.do_flow} {eta_calib_val} "
                 f"{do_rcone_val} {do_neg_energy_threshold_val} {neg_energy_threshold_val} $(zs_cemc) $(zs_hcalin) $(zs_hcalout) {do_emfrac_calib_val} "
                 f"{args.jet_pt_min} {args.truth_jet_pt_min} {manager.output_dir}/output")
    manager.write_submit_file(arguments=arguments)
    manager.finalize_submission(queue_arg="input_dst_global,input_dst_jet,input_dst_g4hit,input_dst_calo,input_dst_zdc,input_dst_sepd,input_calib,zs_cemc,zs_hcalin,zs_hcalout from jobs.list")

def create_f4a_zdc_jobs(args):
    manager = CondorJobManager(args, job_name="F4A ZDC")
    manager.add_file_to_check(args.f4a_macro)
    manager.validate_paths()

    manager.log_initialization({
        'Logging Interval': f"{args.log_interval} Events",
        'Fun4All Macro': Path(args.f4a_macro).resolve()
    })

    files_dir = manager.prepare_directories()
    manager.copy_dependencies(extra_files=[args.f4a_macro])

    jobs_file = manager.output_dir / 'jobs.list'
    jobs_file.unlink(missing_ok=True)

    all_dst_lines = []
    for list_path_str in manager.input_list.read_text(encoding='utf-8').splitlines():
        list_path_str = list_path_str.strip()
        if not list_path_str:
            continue

        sub_list = Path(list_path_str)
        if sub_list.is_file():
            all_dst_lines.extend(sub_list.read_text(encoding='utf-8').splitlines())
        else:
            manager.logger.warning(f"Sub-list file not found: {sub_list}")

    total_jobs = 0
    with open(jobs_file, mode='w', encoding='utf-8') as f_jobs:
        for i, chunk in enumerate(chunk_list(all_dst_lines, args.dst_per_job)):
            chunk_file = files_dir / f'chunk-{i:05d}.list'
            chunk_file.write_text("\n".join(chunk) + "\n", encoding='utf-8')
            f_jobs.write(f"{chunk_file.resolve()}\n")
            total_jobs += 1

    manager.logger.info(f"Total jobs prepared: {total_jobs}")

    arguments = f"{manager.output_dir / Path(args.f4a_macro).name} $(input_dst) {args.events} {args.dbtag} {args.log_interval} {manager.output_dir}/output"
    manager.write_submit_file(arguments=arguments)
    manager.finalize_submission(queue_arg="input_dst from jobs.list")

def create_f4a_sepd_calib_jobs(args):
    manager = CondorJobManager(args, job_name="F4A sEPD Calib")
    manager.add_file_to_check(args.f4a_macro)
    manager.validate_paths()

    manager.log_initialization({
        'Logging Interval': f"{args.log_interval} Events",
        'Fun4All Macro': Path(args.f4a_macro).resolve()
    })

    files_dir = manager.prepare_directories()
    manager.copy_dependencies(extra_files=[args.f4a_macro])

    jobs_file = manager.output_dir / 'jobs.list'
    jobs_file.unlink(missing_ok=True)

    all_dst_lines = []
    lines = [line.strip() for line in manager.input_list.read_text(encoding='utf-8').splitlines() if line.strip()]
    if any(Path(l).is_file() for l in lines):
        for list_path_str in lines:
            sub_list = Path(list_path_str)
            if sub_list.is_file():
                all_dst_lines.extend(sub_list.read_text(encoding='utf-8').splitlines())
            else:
                manager.logger.warning(f"Sub-list file not found: {sub_list}")
    else:
        all_dst_lines = lines

    all_dst_lines = [l.strip() for l in all_dst_lines if l.strip()]

    total_jobs = 0
    with open(jobs_file, mode='w', encoding='utf-8') as f_jobs:
        for i, chunk in enumerate(chunk_list(all_dst_lines, args.dst_per_job)):
            chunk_file = files_dir / f'chunk-{i:05d}.list'
            chunk_file.write_text("\n".join(chunk) + "\n", encoding='utf-8')
            f_jobs.write(f"{chunk_file.resolve()}\n")
            total_jobs += 1

    manager.logger.info(f"Total jobs prepared: {total_jobs}")

    arguments = f"{manager.output_dir / Path(args.f4a_macro).name} $(input_dst) {args.events} {args.dbtag} {args.log_interval} {manager.output_dir}/output"
    manager.write_submit_file(arguments=arguments)
    manager.finalize_submission(queue_arg="input_dst from jobs.list")

create_f4a_sepdcalib_jobs = create_f4a_sepd_calib_jobs

def create_f4a_mc_jobs(args):
    manager = CondorJobManager(args, job_name="F4A MC")
    manager.add_file_to_check(args.f4a_macro)
    manager.add_dir_to_check(args.src_dir)
    manager.validate_paths()

    manager.log_initialization({
        'Fun4All Macro': Path(args.f4a_macro).resolve(),
        'Source Directory': Path(args.src_dir).resolve()
    })

    files_dir = manager.prepare_directories()
    manager.copy_dependencies(extra_files=[args.f4a_macro], extra_dirs=[args.src_dir])

    manager.prepare_job_lists(dst_per_job=args.dst_per_job, files_dir=files_dir)

    arguments = f"{manager.output_dir / Path(args.f4a_macro).name} $(input_dst) test-$(ClusterId)-$(Process).root {args.events} {args.dbtag} {manager.output_dir}/output"
    manager.write_submit_file(arguments=arguments)
    manager.finalize_submission(queue_arg="input_dst from jobs.list")

def create_f4a_data_mc_jobs(args):
    manager = CondorJobManager(args, job_name="F4A Data MC")
    manager.add_file_to_check(args.f4a_macro)
    manager.add_file_to_check(args.calo_calib_macro)
    manager.add_file_to_check(args.HIJetReco_macro)
    manager.add_dir_to_check(args.src_dir)

    calib_list = Path(args.calib).resolve() if args.calib else None
    if calib_list:
        manager.add_file_to_check(calib_list)

    manager.validate_paths()

    manager.log_initialization({
        'Calib List': calib_list if calib_list else "Not Provided (Using default)",
        'Jet pT Min': f"{args.jet_pt_min} GeV",
        'Fun4All Macro': Path(args.f4a_macro).resolve(),
        'Calo Calib Macro': Path(args.calo_calib_macro).resolve(),
        'HIJetReco Macro': Path(args.HIJetReco_macro).resolve(),
        'Source Directory': Path(args.src_dir).resolve()
    })

    files_dir = manager.prepare_directories()

    extra_files = [args.f4a_macro, args.calo_calib_macro, args.HIJetReco_macro]
    if calib_list:
        extra_files.append(calib_list)

    manager.copy_dependencies(extra_files=extra_files, extra_dirs=[args.src_dir])

    calib_map = parse_calib_list(calib_list, manager.logger)

    jobs_file = manager.output_dir / 'jobs.list'
    jobs_file.unlink(missing_ok=True)

    with open(manager.input_list, mode='r', encoding='utf-8') as file_in, \
         open(jobs_file, mode='w', encoding='utf-8') as file_out:
        for line in file_in:
            line = line.strip()
            if not line:
                continue

            if not calib_map:
                file_out.write(f'{line},default\n')
            else:
                line_path = Path(line).resolve()
                try:
                    run = line_path.stem.split('-')[3].lstrip('0')
                    if run in calib_map:
                        file_out.write(f'{line},{calib_map[run]}\n')
                    else:
                        file_out.write(f'{line},default\n')
                except IndexError:
                    manager.logger.warning(f"Could not parse run number from {line}. Using default.")
                    file_out.write(f'{line},default\n')

    arguments = f"{manager.output_dir / Path(args.f4a_macro).name} $(input_dst_global) $(input_dst_jet) $(input_dst_g4hit) $(input_dst_calo) $(input_calib) test-$(ClusterId)-$(Process).root tree-$(ClusterId)-$(Process).root {args.events} {args.jet_pt_min} {args.dbtag} {manager.output_dir}/output"
    manager.write_submit_file(arguments=arguments)
    manager.finalize_submission(queue_arg="input_dst_global,input_dst_jet,input_dst_g4hit,input_dst_calo,input_calib from jobs.list")

def create_f4a_nobkgsub_jobs(args):
    manager = CondorJobManager(args, job_name="F4A NoBkgSub")
    manager.add_file_to_check(args.f4a_macro)
    manager.add_file_to_check(args.calo_calib_macro)
    manager.add_file_to_check(args.NoBkgdSubJetReco_macro)
    manager.add_dir_to_check(args.src_dir)
    manager.validate_paths()

    manager.log_initialization({
        'Fun4All Macro': Path(args.f4a_macro).resolve(),
        'Calo Calib Macro': Path(args.calo_calib_macro).resolve(),
        'NoBkgdSubJetReco Macro': Path(args.NoBkgdSubJetReco_macro).resolve(),
        'Source Directory': Path(args.src_dir).resolve()
    })

    files_dir = manager.prepare_directories()
    extra_files = [args.f4a_macro, args.calo_calib_macro, args.NoBkgdSubJetReco_macro]
    manager.copy_dependencies(extra_files=extra_files, extra_dirs=[args.src_dir])

    manager.prepare_job_lists(dst_per_job=args.dst_per_job, files_dir=files_dir)

    arguments = f"{manager.output_dir / Path(args.f4a_macro).name} $(input_dst) test-$(ClusterId)-$(Process).root tree-$(ClusterId)-$(Process).root {args.events} {args.dbtag} {manager.output_dir}/output"
    manager.write_submit_file(arguments=arguments)
    manager.finalize_submission(queue_arg="input_dst from jobs.list")

def create_f4a_noise_jobs(args):
    manager = CondorJobManager(args, job_name="F4A Noise")
    manager.add_file_to_check(args.f4a_macro)
    manager.add_dir_to_check(args.src_dir)
    manager.validate_paths()

    manager.log_initialization({
        'Fun4All Macro': Path(args.f4a_macro).resolve(),
        'Source Directory': Path(args.src_dir).resolve()
    })

    files_dir = manager.prepare_directories()
    manager.copy_dependencies(extra_files=[args.f4a_macro], extra_dirs=[args.src_dir])

    manager.prepare_job_lists(dst_per_job=args.dst_per_job, files_dir=files_dir)

    arguments = f"{manager.output_dir / Path(args.f4a_macro).name} $(input_dst) test-$(ClusterId)-$(Process).root {args.events} {args.dbtag} {manager.output_dir}/output"
    manager.write_submit_file(arguments=arguments)
    manager.finalize_submission(queue_arg="input_dst from jobs.list")

def create_f4a_calofittingqa_jobs(args):
    manager = CondorJobManager(args, job_name="F4A CaloFittingQA")
    manager.add_file_to_check(args.f4a_macro)
    manager.validate_paths()

    manager.log_initialization({
        'Fun4All Macro': Path(args.f4a_macro).resolve()
    })

    files_dir = manager.prepare_directories()
    manager.copy_dependencies(extra_files=[args.f4a_macro])

    manager.prepare_job_lists(dst_per_job=args.dst_per_job, files_dir=files_dir)

    arguments = f"{manager.output_dir / Path(args.f4a_macro).name} $(input_dst) test-$(ClusterId)-$(Process).root {args.events} {args.dbtag} {manager.output_dir}/output"
    manager.write_submit_file(arguments=arguments)
    manager.finalize_submission(queue_arg="input_dst from jobs.list")

def create_f4a_sepdqa_jobs(args):
    manager = CondorJobManager(args, job_name="F4A sEPDQA")
    manager.add_file_to_check(args.f4a_macro)
    manager.add_file_to_check(args.calo_calib_macro)
    manager.add_dir_to_check(args.src_dir)
    manager.validate_paths()

    manager.log_initialization({
        'Fun4All Macro': Path(args.f4a_macro).resolve(),
        'Calo Calib Macro': Path(args.calo_calib_macro).resolve(),
        'Source Directory': Path(args.src_dir).resolve()
    })

    files_dir = manager.prepare_directories()
    manager.copy_dependencies(extra_files=[args.f4a_macro, args.calo_calib_macro], extra_dirs=[args.src_dir])

    manager.prepare_job_lists(dst_per_job=args.dst_per_job, files_dir=files_dir)

    arguments = f"{manager.output_dir / Path(args.f4a_macro).name} $(input_dst) tree-$(ClusterId)-$(Process).root {args.events} {args.dbtag} {manager.output_dir}/output"
    manager.write_submit_file(arguments=arguments)
    manager.finalize_submission(queue_arg="input_dst from jobs.list")

def setup_f4a_subparsers(subparsers):
    f4a = subparsers.add_parser('f4a', parents=[get_common_parser()], help='Create condor submission directory.')
    add_build_args(f4a)
    f4a.add_argument('-i2_calib', '--calib', type=str, default=None, help='Q Vector Calibrations. (Optional)')
    f4a.add_argument('-i3_calib', '--eta-calib-path', '--eta-calib-direct-path', dest='eta_calib_path', type=str, default=None, help='Direct path to eta-shape calibration file. (Optional)')
    f4a.add_argument('-i4_calib', '--zs-calib', dest='zs_calib', type=str, default=None, help='List of ZS cross calib T0 files (CEMC/HCALIN/HCALOUT per run), used by Fun4All_BkgSub. Runs without an entry fall back to the CDB. (Optional)')
    f4a.add_argument('-el', '--event-list', dest='event_list', type=str, default=None, help='Direct path to event list file: one event ID per line (used by every job), or a run,event CSV (header "run,event", Fun4All_BkgSub only) that is split per job by segment = event // 100000, dropping jobs without listed events. (Optional)')
    f4a.add_argument('-f', '--f4a-macro', type=str, default='macros/Fun4All_sEPD.C', help='Fun4All Macro.')
    f4a.add_argument('-f5', '--calo-calib-macro', type=str, default='macros/Calo_Calib.C', help='Calo_Calib Macro.')
    f4a.add_argument('-f6', '--HIJetReco-macro', type=str, default='macros/HIJetReco.C', help='HIJetReco Macro.')
    f4a.add_argument('--do-flow', type=int, default=3, help='Flow modulation configuration for JetReco/QA (default=3)')
    f4a.add_argument('--do-rcone', action='store_true', default=False, help='Enable random cone validation.')
    f4a.add_argument('--do-mult', type=int, default=None, choices=[0, 1], help='Enable multiplicity subtraction (default: 1 if --eta-calib-path is given, else 0).')
    f4a.add_argument('--do-neg-energy-threshold', type=int, default=1, choices=[0, 1], help='Enable negative energy threshold rejection (default=1).')
    f4a.add_argument('--neg-energy-threshold', type=float, default=-2.0, help='Negative energy threshold cut in GeV (default=-2.0).')
    f4a.add_argument('--do-emfrac-calib', type=int, default=0, choices=[0, 1], help='Also apply the EMfrac JES alongside the legacy one and store pt_calib_emfrac / calo fractions (default=0).')
    f4a.add_argument('--overlay', action='store_true', default=False, help='Data+MC overlay mode of Fun4All_BkgSub: -i is an overlay list (DST_GLOBAL,DST_TRUTH_JET,DST_TRUTH_G4HIT,DST_CALO per line), one job per line. Only the iterative jets (with and without flow) plus truth jets are stored. Needs --data-list.')
    f4a.add_argument('--data-list', dest='data_list', type=str, default=None, help='Data list(s) with calofit,zdc,sepd per line (e.g. dst_calofitting_zdc_sepd-00068144.list, or a list of such lists), used with --overlay for the ZDC/sEPD DSTs of each segment.')
    f4a.add_argument('--jet-pt-min', type=float, default=10, help='Reco jet calibrated pT min cut, used with --overlay. Default: 10 GeV.')
    f4a.add_argument('--truth-jet-pt-min', type=float, default=5, help='Truth jet pT min cut, used with --overlay. Default: 5 GeV.')
    f4a.set_defaults(memory=1.5, condor_script='scripts/genFun4All.sh', func=create_f4a_jobs)

    f4a_zdc = subparsers.add_parser('f4a_zdc', parents=[get_common_parser()], help='Create condor submission directory.')
    add_build_args(f4a_zdc, myinstall=False)
    f4a_zdc.add_argument('-n3', '--log-interval', type=int, default=10000, help='Logging Event Frequency. Default: 10000.')
    f4a_zdc.add_argument('-f', '--f4a-macro', type=str, default='macros/Fun4All_ZDC.C', help='Fun4All Macro.')
    f4a_zdc.set_defaults(memory=0.5, condor_script='scripts/genFun4AllZDC.sh', func=create_f4a_zdc_jobs)

    f4a_sepd_calib = subparsers.add_parser('f4a_sepd_calib', aliases=['f4a_sepdcalib'], parents=[get_common_parser()], help='Create condor submission directory for sEPD Calibration.')
    add_build_args(f4a_sepd_calib, myinstall=False)
    f4a_sepd_calib.add_argument('-n3', '--log-interval', type=int, default=5000, help='Logging Event Frequency. Default: 5000.')
    f4a_sepd_calib.add_argument('-f', '--f4a-macro', type=str, default='macros/Fun4All_sEPD_Calib.C', help='Fun4All Macro.')
    f4a_sepd_calib.set_defaults(memory=0.5, condor_script='scripts/genFun4All_sEPD_Calib.sh', func=create_f4a_sepd_calib_jobs)

    f4a_mc = subparsers.add_parser('f4a_mc', parents=[get_common_parser()], help='Create condor submission directory.')
    add_build_args(f4a_mc)
    f4a_mc.add_argument('-f', '--f4a-macro', type=str, default='macros/Fun4All_sEPD_MC.C', help='Fun4All Macro.')
    f4a_mc.set_defaults(dbtag='MDC2', dst_per_job=500, memory=0.5, condor_script='scripts/genFun4All_MC.sh', func=create_f4a_mc_jobs)

    f4a_data_mc = subparsers.add_parser('f4a_data_mc', parents=[get_common_parser()], help='Create condor submission directory.')
    add_build_args(f4a_data_mc)
    f4a_data_mc.add_argument('-i2_calib', '--calib', type=str, default=None, help='Q Vector Calibrations. (Optional)')
    f4a_data_mc.add_argument('-c', '--jet-pt-min', type=float, default=10, help='Jet pT min cut. Default: 10 GeV.')
    f4a_data_mc.add_argument('-f', '--f4a-macro', type=str, default='macros/Fun4All_sEPD_DataMC.C', help='Fun4All Macro.')
    f4a_data_mc.add_argument('-f5', '--calo-calib-macro', type=str, default='macros/Calo_Calib.C', help='Calo_Calib Macro.')
    f4a_data_mc.add_argument('-f6', '--HIJetReco-macro', type=str, default='macros/HIJetReco.C', help='HIJetReco Macro.')
    f4a_data_mc.set_defaults(memory=0.5, condor_script='scripts/genFun4All_DataMC.sh', func=create_f4a_data_mc_jobs)

    f4a_nobkgsub = subparsers.add_parser('f4a_nobkgsub', parents=[get_common_parser()], help='Create condor submission directory for NoBkgSub.')
    add_build_args(f4a_nobkgsub)
    f4a_nobkgsub.add_argument('-f', '--f4a-macro', type=str, default='macros/Fun4All_NoBkgSub.C', help='Fun4All Macro.')
    f4a_nobkgsub.add_argument('-f5', '--calo-calib-macro', type=str, default='macros/Calo_Calib.C', help='Calo_Calib Macro.')
    f4a_nobkgsub.add_argument('-f6', '--NoBkgdSubJetReco-macro', type=str, default='macros/NoBkgdSubJetReco.C', help='NoBkgdSubJetReco Macro.')
    f4a_nobkgsub.set_defaults(memory=1.5, condor_script='scripts/genFun4All_NoBkgSub.sh', func=create_f4a_nobkgsub_jobs)

    f4a_noise = subparsers.add_parser('f4a_noise', parents=[get_common_parser()], help='Create condor submission directory for Noise.')
    add_build_args(f4a_noise)
    f4a_noise.add_argument('-f', '--f4a-macro', type=str, default='macros/Fun4All_Noise.C', help='Fun4All Macro.')
    f4a_noise.set_defaults(memory=1.5, condor_script='scripts/genFun4All_Noise.sh', func=create_f4a_noise_jobs)

    f4a_calofittingqa = subparsers.add_parser('f4a_calofittingqa', parents=[get_common_parser()], help='Create condor submission directory for CaloFittingQA.')
    add_build_args(f4a_calofittingqa)
    f4a_calofittingqa.add_argument('-f', '--f4a-macro', type=str, default='macros/Fun4All_sEPD_CaloFittingQA.C', help='Fun4All Macro.')
    f4a_calofittingqa.set_defaults(memory=1.5, condor_script='scripts/genFun4All_CaloFittingQA.sh', func=create_f4a_calofittingqa_jobs)

    f4a_sepdqa = subparsers.add_parser('f4a_sepdqa', parents=[get_common_parser()], help='Create condor submission directory for sEPDQA.')
    add_build_args(f4a_sepdqa)
    f4a_sepdqa.add_argument('-f', '--f4a-macro', type=str, default='macros/Fun4All_sEPDQA.C', help='Fun4All Macro.')
    f4a_sepdqa.add_argument('-f5', '--calo-calib-macro', type=str, default='macros/Calo_Calib.C', help='Calo_Calib Macro.')
    f4a_sepdqa.set_defaults(memory=1.5, condor_script='scripts/genFun4All_sEPDQA.sh', func=create_f4a_sepdqa_jobs)
