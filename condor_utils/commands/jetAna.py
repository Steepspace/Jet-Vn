import argparse
import re
import sys
from pathlib import Path

from condor_utils.core.helpers import chunk_list
from condor_utils.core.manager import CondorJobManager
from condor_utils.cli import get_common_parser, add_build_args

# Leading truth R=0.4 jet pT window [min, max) of each data+MC overlay pT-hat sample, max < 0: no upper edge.
# Each window starts where the sample is complete: luminosity normalized, it agrees with the next lower sample
# from there on. The generator trigger turn-on extends past the nominal threshold (run 68144 overlay):
# Jet20 is 12% low at 20-21 GeV and 2.5% low up to 25 GeV, Jet30 / Jet40 are 7-8% low in their first GeV.
OVERLAY_WINDOWS = {"Jet12": (12.0, 25.0), "Jet20": (25.0, 31.0), "Jet30": (31.0, 41.0), "Jet40": (41.0, -1.0)}

# PYTHIA8 embedded in HIJING (MDC2 run 31, no Jet40): provisional windows at the nominal thresholds, to be set
# from the luminosity normalized leading truth R=0.4 jet spectra (hMaxPtTruthR04_all, filled before any window).
HIJING_WINDOWS = {"Jet10": (10.0, 20.0), "Jet20": (20.0, 30.0), "Jet30": (30.0, -1.0)}

# Jet-JES mode (9th argument): 0 data+MC overlay with flow (iter / iternf), 1 p+p MC (unsub / iternf),
# 2 Au+Au without flow (HIJING embedded MC or no-flow overlay, iternf), 3 minimum-bias HIJING without truth (iternf,
# OHCal fraction templates only)
JES_MODE_OVERLAY, JES_MODE_PP, JES_MODE_NOFLOW, JES_MODE_MB = 0, 1, 2, 3

# minimum-bias HIJING trees (<outdir>/MB/tree/...): one sample, no truth and so no window (the range is not used)
MB_SAMPLE = "MB"
MB_WINDOW = (0.0, -1.0)

def jes_mode(args):
    """Jet-JES mode of the jetJES options."""
    if args.pp:
        return JES_MODE_PP
    if args.mb:
        return JES_MODE_MB
    if args.hijing or args.noflow:
        return JES_MODE_NOFLOW
    return JES_MODE_OVERLAY

def jes_mode_label(args):
    if args.pp:
        return "p+p MC (unsub / iternf)"
    if args.mb:
        return "minimum-bias HIJING without truth (iternf, OHCal fraction templates only, no Calo-MBD cut)"
    if args.hijing:
        return "PYTHIA8 embedded in HIJING MC (iternf, no Calo-MBD cut)"
    if args.noflow:
        return "data+MC overlay without flow (iternf)"
    return "data+MC overlay with flow (iter / iternf)"

def parse_window(text):
    """Parses SAMPLE=MIN,MAX (e.g. Jet20=20,30 or Jet40=40,-1) into (sample, (min, max))."""
    m = re.fullmatch(r'(\w+)=([^,]+),([^,]+)', text)
    try:
        if not m:
            raise ValueError
        return m.group(1), (float(m.group(2)), float(m.group(3)))
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid window '{text}', expected SAMPLE=MIN,MAX (e.g. Jet20=20,30)") from None

def run_jet_jobs(args, version):
    manager = CondorJobManager(args, job_name=version)
    if version == "jetAnav3":
        manager.add_file_to_check(args.calo_mbd_list)
        if args.jes_z0_file:
            manager.add_file_to_check(args.jes_z0_file)
        if bool(args.timing_fc_file) != bool(args.timing_lines_file):
            manager.logger.critical("The timing correction needs both --timing-fc-file and --timing-lines-file")
            sys.exit(1)
        for timing_file in (args.timing_fc_file, args.timing_lines_file):
            if timing_file:
                manager.add_file_to_check(timing_file)
    elif version == "jetDijet":
        manager.add_file_to_check(args.calo_mbd_list)
        if args.timing_corr_file:
            manager.add_file_to_check(args.timing_corr_file)
    elif version == "jetJES":
        # MC without a data run (p+p, HIJING embedded, MB HIJING) has no Calo-MBD selection
        mc_only = args.pp or args.hijing or args.mb
        if not mc_only:
            if not args.calo_mbd_list:
                manager.logger.critical("jetJES needs --calo-mbd-list (or --pp / --hijing / --mb for MC trees without a data run)")
                sys.exit(1)
            manager.add_file_to_check(args.calo_mbd_list)
    manager.add_file_to_check(args.jetAna_macro)
    manager.add_file_to_check(args.jetAna_bin)

    manager.validate_paths()

    jet_pt_min      = args.jet_pt_min
    files_per_job   = args.files_per_job

    init_log = {
        'Jet pT Min': f"{jet_pt_min} GeV",
        'Files Per Job': files_per_job,
        'Jet Ana Macro': Path(args.jetAna_macro).resolve(),
        'Jet Ana Bin': Path(args.jetAna_bin).resolve(),
    }

    if version == "jetAnav3":
        init_log['Calo-MBD List'] = Path(args.calo_mbd_list).resolve()
        init_log['Do Iter'] = args.do_iter
        init_log['Do Mult'] = args.do_mult
        init_log['Do Unsub'] = args.do_unsub
        init_log['Do RCone'] = args.do_rcone
        init_log['Lead Jet pT Threshold'] = f"{args.lead_jet_pt_threshold} GeV"
        init_log['Sigma Cut'] = args.sigma_cut
        init_log['Iter Is NoFlow'] = args.iter_is_noflow
        init_log['JES Z0 File'] = Path(args.jes_z0_file).resolve() if args.jes_z0_file else "Not Provided (Using stored eta-binned pt_calib)"
        init_log['Timing f_c File'] = Path(args.timing_fc_file).resolve() if args.timing_fc_file else "Not Provided (no timing correction)"
        init_log['Timing Lines File'] = Path(args.timing_lines_file).resolve() if args.timing_lines_file else "Not Provided (no timing correction)"
    elif version == "jetDijet":
        # the jets are cut in raw pT only
        del init_log['Jet pT Min']
        init_log['Calo-MBD List'] = Path(args.calo_mbd_list).resolve()
        init_log['Jet Raw pT Min'] = f"{args.jet_pt_raw_min} GeV"
        init_log['Sigma Cut'] = args.sigma_cut
        init_log['Dijet pT_sub Min'] = f"R = 0.2: {args.pt_sub_min_r02} GeV, R = 0.3: {args.pt_sub_min_r03} GeV (raw)"
        init_log['Timing Correction File'] = Path(args.timing_corr_file).resolve() if args.timing_corr_file else "Not Provided (no corrected timing hists)"
    elif version == "jetJES":
        # the jet pT is not cut on, the response is filled for every matched truth jet
        del init_log['Jet pT Min']
        init_log['Mode'] = f"{jes_mode(args)}: {jes_mode_label(args)}"
        if not mc_only:
            init_log['Calo-MBD List'] = Path(args.calo_mbd_list).resolve()
        if args.mb:
            windows = {MB_SAMPLE: MB_WINDOW}
            init_log['Lead Truth R=0.4 Jet pT Windows'] = "none (no truth)"
        else:
            windows = {**(HIJING_WINDOWS if args.hijing else OVERLAY_WINDOWS), **dict(args.window or [])}
            init_log['Lead Truth R=0.4 Jet pT Windows'] = ", ".join(
                f"{sample}: [{lo}, {'inf' if hi < 0 else hi})" for sample, (lo, hi) in windows.items()) + " GeV"
        if not mc_only:
            init_log['Sigma Cut'] = args.sigma_cut
        init_log['Failed-Match Dump'] = f"truth pT >= {args.dump_pt_min} GeV" if args.dump_pt_min > 0 else "off"

    manager.log_initialization(init_log)

    files_dir = manager.prepare_directories()

    manager.copy_dependencies(extra_files=[args.jetAna_macro, args.jetAna_bin])

    run_trees = {}
    input_lines = manager.input_list.read_text(encoding='utf-8').splitlines()

    for line in input_lines:
        line_str = line.strip()
        if not line_str:
            continue
        tree_path = Path(line_str)
        run_id = tree_path.parent.parent.name
        if not run_id.isdigit():
            m = re.search(r'/(?:run_)?(\d{5,8})(?:/|$)', str(tree_path))
            if m:
                run_id = m.group(1)
        if run_id not in run_trees:
            run_trees[run_id] = []
        run_trees[run_id].append(str(tree_path))

    # job groups: (run, chunk file tag, trees, extra jobs.list fields)
    if version == "jetJES":
        # the overlay pT-hat samples share the data run number: split each run by sample,
        # every job gets the window of its sample and writes to its own sample directory.
        # p+p and HIJING MC trees have no data run (<sample>/tree/...): grouped by sample only, the sample is the tag
        mc_tag = "pp" if args.pp else ("mb" if args.mb else "hijing")
        sample_trees = {}
        for run_id, trees in run_trees.items():
            for tree in trees:
                # MB HIJING: a single sample, whatever the path
                if args.mb:
                    sample_trees.setdefault((mc_tag, MB_SAMPLE), []).append(tree)
                    continue
                # overlay paths have /JetNN/, the p+p / HIJING MC paths /jetNN/
                m = re.search(r'/[Jj]et(\d+)/', tree)
                if not m:
                    manager.logger.critical(f"No pT-hat sample (/JetNN/ or /jetNN/) in tree path: {tree}")
                    sys.exit(1)
                sample = f"Jet{m.group(1)}"
                sample_trees.setdefault((mc_tag if mc_only else run_id, sample), []).append(tree)

        missing = sorted({sample for _, sample in sample_trees} - windows.keys())
        if missing:
            manager.logger.critical(f"No lead truth pT window for samples {missing}: add --window SAMPLE=MIN,MAX")
            sys.exit(1)

        groups = []
        for (run_id, sample), trees in sorted(sample_trees.items()):
            lo, hi = windows[sample]
            if mc_only:
                if args.mb:
                    manager.logger.info(f"MB HIJING Sample {sample}: {len(trees)} trees, no truth (no window)")
                else:
                    manager.logger.info(f"{'p+p' if args.pp else 'HIJING'} MC Sample {sample}: {len(trees)} trees, window [{lo}, {'inf' if hi < 0 else hi}) GeV")
                groups.append((run_id, sample, trees, f",{lo},{hi},{sample}"))
            else:
                manager.logger.info(f"Overlay Sample {sample} (run {run_id}): {len(trees)} trees, window [{lo}, {'inf' if hi < 0 else hi}) GeV")
                groups.append((run_id, f"{sample}_{run_id}", trees, f",{lo},{hi},{sample}"))
    else:
        groups = [(run_id, run_id, trees, "") for run_id, trees in run_trees.items()]

    if version == "jetJES" and mc_only:
        run_map = {}
    else:
        run_paths = [Path(l.strip()) for l in Path(args.calo_mbd_list).resolve().read_text(encoding='utf-8').splitlines() if l.strip()]
        run_map = {}
        for p in run_paths:
            run_map[p.stem] = str(p)
            run_map[p.stem.lstrip('0')] = str(p)

    jobs_list_file = manager.output_dir / 'jobs.list'
    jobs_list_file.unlink(missing_ok=True)
    total_jobs = 0

    with open(jobs_list_file, mode='w', encoding='utf-8') as f_jobs:
        for run_id, tag, trees, extra in groups:
            if version == "jetJES" and mc_only:
                qa_file = "none"
            else:
                qa_file = run_map.get(run_id) or run_map.get(run_id.lstrip('0'))
                if not qa_file:
                    manager.logger.warning(f"Run {run_id} found in input trees but not in Calo-MBD list. Skipping.")
                    continue

            for i, chunk in enumerate(chunk_list(trees, files_per_job)):
                chunk_file = files_dir / f'{tag}_part_{i}.list'
                chunk_file.write_text("\n".join(chunk) + "\n", encoding='utf-8')

                f_jobs.write(f"{chunk_file},{qa_file}{extra}\n")
                total_jobs += 1

    manager.logger.info(f"Total jobs prepared: {total_jobs}")

    if version == "jetAnav3":
        # the timing correction files are read in place by the jobs: pass absolute paths
        timing_files = " ".join(str(Path(f).resolve()) if f else "none" for f in (args.timing_fc_file, args.timing_lines_file))
        arguments = f"{manager.output_dir / Path(args.jetAna_bin).name} $(input_tree_list) $(input_calo_mbd) {jet_pt_min} {manager.output_dir}/output {args.do_iter} {args.do_mult} {args.do_unsub} {args.do_rcone} {args.lead_jet_pt_threshold} {args.sigma_cut} {args.jes_z0_file or 'none'} {args.iter_is_noflow} {timing_files}"
        queue_arg = "input_tree_list,input_calo_mbd from jobs.list"
    elif version == "jetDijet":
        # the timing correction file is read in place by the jobs: pass its absolute path
        timing_corr = str(Path(args.timing_corr_file).resolve()) if args.timing_corr_file else "none"
        arguments = f"{manager.output_dir / Path(args.jetAna_bin).name} $(input_tree_list) $(input_calo_mbd) {manager.output_dir}/output {args.jet_pt_raw_min} {args.sigma_cut} {args.pt_sub_min_r02} {args.pt_sub_min_r03} {timing_corr}"
        queue_arg = "input_tree_list,input_calo_mbd from jobs.list"
    elif version == "jetJES":
        arguments = f"{manager.output_dir / Path(args.jetAna_bin).name} $(input_tree_list) $(input_calo_mbd) {manager.output_dir}/output $(lead_truth_pt_min) $(lead_truth_pt_max) {args.sigma_cut} $(sample) {jes_mode(args)} {args.dump_pt_min}"
        queue_arg = "input_tree_list,input_calo_mbd,lead_truth_pt_min,lead_truth_pt_max,sample from jobs.list"

    sub_file_name = f"{manager.condor_script.stem}.sub"
    manager.write_submit_file(arguments=arguments, sub_file_name=sub_file_name)
    manager.finalize_submission(queue_arg=queue_arg, sub_file_name=sub_file_name)

def jetAnav3_jobs(args):
    run_jet_jobs(args, "jetAnav3")

def jetDijet_jobs(args):
    run_jet_jobs(args, "jetDijet")

def jetJES_jobs(args):
    run_jet_jobs(args, "jetJES")

def add_common_args(parser, name_suffix):
    parser.add_argument('-i', '--input-list', type=str, required=True, help='List of TTrees to analyze.')
    parser.add_argument('-j', '--jet-pt-min', type=float, default=10, help='Minimum Jet pT. Default: 10 [GeV]')
    parser.add_argument('-p', '--files-per-job', type=int, default=100, help='Number of trees per job list. Default: 100')
    parser.add_argument('-f', '--jetAna-macro', type=str, default=f'macros/analysis/Jet-{name_suffix}.C', help=f'Jet-Ana Macro. Default: macros/analysis/Jet-{name_suffix}.C')
    parser.add_argument('-f2', '--jetAna-bin', type=str, default=f'bin/Jet-{name_suffix}', help=f'Jet-Ana Bin. Default: bin/Jet-{name_suffix}')
    parser.add_argument('-o', '--output-dir', type=str, default='scratch/test', help='Output Directory. Default: scratch/test')
    parser.add_argument('-s', '--memory', type=float, default=1.5, help='Memory (units of GB). Default: 1.5 GB.')
    parser.add_argument('--retry-memory-step', type=float, default=0.5, help='Memory in GB to add if job is evicted for exceeding memory. Default: 0.5 GB. Set to 0 to disable.')
    parser.add_argument('--retry-memory-max', type=float, default=6.0, help='Ceiling / max memory in GB for automatic memory retries. Default: 6.0 GB.')
    parser.add_argument('--retry-request-memory', type=str, default=None, help='Explicit retry_request_memory setting (e.g. "1.5GB"). If None, computed as memory + retry_memory_step up to retry_memory_max.')
    parser.add_argument('-m', '--max-retries', type=int, default=3, help='Max Condor job retries on failure. Default: 3.')
    parser.add_argument('-l', '--condor-log-dir', type=str, default='/tmp/anarde/dump', help='Condor Log Directory.')
    parser.add_argument('-f3', '--condor-script', type=str, default=f'scripts/genJet{name_suffix}.sh', help='Condor Script.')
    parser.add_argument('-f4', '--common-errors', type=str, default='files/common-errors.txt', help='Common Errors.')

def setup_jetAna_subparsers(subparsers):
    jetAnav3 = subparsers.add_parser('jetAnav3', help='jetAnav3 condor jobs.')
    add_common_args(jetAnav3, "Anav3")
    add_build_args(jetAnav3)
    jetAnav3.add_argument('-i2', '--calo-mbd-list', '--event-qa-list', dest='calo_mbd_list', type=str, required=True, help='List of Calo-MBD / Event QA files.')
    jetAnav3.add_argument('--do-iter', type=int, default=1, help='Do iter. Default: 1')
    jetAnav3.add_argument('--do-mult', type=int, default=0, help='Do mult. Default: 0')
    jetAnav3.add_argument('--do-unsub', type=int, default=1, help='Do unsub. Default: 1')
    jetAnav3.add_argument('--do-rcone', type=int, default=0, help='Do rcone. Default: 0')
    jetAnav3.add_argument('--lead-jet-pt-threshold', type=float, default=100.0, help='Leading Jet pT Threshold for CSV saving. Default: 100 [GeV]')
    jetAnav3.add_argument('--sigma-cut', type=float, default=3.5, help='Calo-MBD consistency cut in sigma. Default: 3.5')
    jetAnav3.add_argument('--jes-z0-file', type=str, default=None, help='Legacy JES payload (JES_Calib_Default); recalibrates the raw jet pT with its eta-averaged R0X_Z0 function instead of the stored eta-binned pt_calib. (Optional)')
    jetAnav3.add_argument('--iter-is-noflow', type=int, default=1, help='Treat the stored iter branches as the no-flow (iternf) version and ignore flow inputs (psi2, calo_v2, flow failure, mult). Default: 1')
    jetAnav3.add_argument('--timing-fc-file', type=str, default=None, help='Leading jet timing correction: f_c(x) per radius and centrality class (timing_correction_iternf.csv of plot_jet_iternf_timing.py). With --timing-lines-file, fills the *_corr timing hists. (Optional)')
    jetAnav3.add_argument('--timing-lines-file', type=str, default=None, help='Leading jet timing correction: per-run lines a_run, b_run (per_run_lines_r03_iternf.csv of plot_jet_iternf_timing.py). (Optional)')
    jetAnav3.set_defaults(func=jetAnav3_jobs)

    jetDijet = subparsers.add_parser('jetDijet', help='jetDijet condor jobs (clean dijet control sample, iternf R = 0.2 / 0.3).')
    add_common_args(jetDijet, "Dijet")
    add_build_args(jetDijet)
    jetDijet.add_argument('-i2', '--calo-mbd-list', '--event-qa-list', dest='calo_mbd_list', type=str, required=True, help='List of Calo-MBD / Event QA files.')
    jetDijet.add_argument('--jet-pt-raw-min', type=float, default=7.0, help='Minimum raw jet pT, where the tree sample (calibrated pT >= 10 GeV) is complete. Default: 7 [GeV]')
    jetDijet.add_argument('--sigma-cut', type=float, default=3.5, help='Calo-MBD consistency cut in sigma. Default: 3.5')
    jetDijet.add_argument('--pt-sub-min-r02', type=float, default=14.25, help='Dijet selection of the timing and jet pT hists, R = 0.2: raw pT_sub >= this (the 0-20%% cut of the purity scan). Default: 14.25 [GeV]')
    jetDijet.add_argument('--pt-sub-min-r03', type=float, default=21.5, help='Dijet selection of the timing and jet pT hists, R = 0.3: raw pT_sub >= this (the 0-20%% cut of the purity scan). Default: 21.5 [GeV]')
    jetDijet.add_argument('--timing-corr-file', type=str, default=None, help='Leading jet timing correction: f(x) per radius and centrality range, fitted to the dijet sample (files/calib/timing_dijet_fit_iternf.csv, from plot_jet_dijet_timing.py --write-calib). Fills the h2LeadTimeOHCalFrac_<r>_<dijet|incl>_*_corr hists and the corrected time vs pT hists h2LeadTimeCorrPt_<r>_<dijet|lead>_* and h2JetTimeCorrPt_<r>_<dijet|all>_*. (Optional)')
    jetDijet.set_defaults(func=jetDijet_jobs)

    jetJES = subparsers.add_parser('jetJES', help='jetJES condor jobs (data+MC overlay, p+p MC or HIJING embedded MC JES, all pT-hat samples of the input list in one submission).')
    add_common_args(jetJES, "JES")
    add_build_args(jetJES)
    jetJES.add_argument('-i2', '--calo-mbd-list', '--event-qa-list', dest='calo_mbd_list', type=str, default=None, help='List of Calo-MBD / Event QA files. Required unless --pp or --hijing.')
    jes_modes = jetJES.add_mutually_exclusive_group()
    jes_modes.add_argument('--pp', action='store_true', default=False, help='p+p MC trees (Fun4All_PP_JES): unsub / iternf jets (the trees\' no-flow iter), single centrality class, no Calo-MBD list.')
    jes_modes.add_argument('--hijing', action='store_true', default=False, help='PYTHIA8 embedded in HIJING trees (Fun4All_HIJING_JES): iternf jets (the trees\' no-flow iter), centrality classes, no Calo-MBD list, HIJING windows.')
    jes_modes.add_argument('--noflow', action='store_true', default=False, help='Data+MC overlay trees made with --do-flow 0: iternf jets (the trees\' no-flow iter), centrality and Calo-MBD selection, no flow / calo v2 cuts.')
    jes_modes.add_argument('--mb', action='store_true', default=False, help='Minimum-bias HIJING trees without truth jets (Fun4All_HIJING_JES with no dst_truth_jet.list): iternf jets, centrality classes, no window, no matching; only the OHCal fraction templates (tasks/jet-ohcal-frac-templates.md). No Calo-MBD list.')
    windows_help = ", ".join(f"{s}={lo:g},{hi:g}" for s, (lo, hi) in OVERLAY_WINDOWS.items())
    hijing_windows_help = ", ".join(f"{s}={lo:g},{hi:g}" for s, (lo, hi) in HIJING_WINDOWS.items())
    jetJES.add_argument('--window', type=parse_window, action='append', metavar='SAMPLE=MIN,MAX', help=f'Leading truth R=0.4 jet pT window [MIN, MAX) GeV of a pT-hat sample (MAX < 0: no upper edge), overrides or adds to the defaults: {windows_help} (--hijing: {hijing_windows_help}, provisional). Repeatable.')
    jetJES.add_argument('--sigma-cut', type=float, default=3.5, help='Calo-MBD consistency cut in sigma. Default: 3.5')
    jetJES.add_argument('--dump-pt-min', type=float, default=-1, help='Write every fiducial truth jet with pT >= this [GeV] that fails the matching, with the event\'s jets and vertices (z_MBD, z_truth if in the trees), to <sample>/Jet-JES_<input>_failed_matches.txt. Default: -1 (off).')
    jetJES.set_defaults(func=jetJES_jobs)
