import argparse
import os
from pathlib import Path

def add_build_args(parser, myinstall=True, short_flag=True):
    """
    Adds --build (and --myinstall, for workers that source setup_local.sh) to a subparser.
    short_flag=False skips -b for commands that already use it.
    """
    default_build = Path(os.environ['OFFLINE_MAIN']).name if os.environ.get('OFFLINE_MAIN') else None
    flags = ['-b', '--build'] if short_flag else ['--build']
    parser.add_argument(*flags, dest='build', type=str, default=default_build, required=default_build is None,
                        help=f'sPHENIX build tag/version (e.g. new, ana.576), resolved to the concrete build at submit time. Default: "{default_build}" (from $OFFLINE_MAIN).')
    if myinstall:
        parser.add_argument('--myinstall', '--install-dir', dest='myinstall', type=str, default=os.environ.get('MYINSTALL'),
                            help='Local install directory sourced via setup_local.sh, or "none". Default: $MYINSTALL (none if unset).')

def get_common_parser():
    """Returns a parent parser with arguments common to most jobs."""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('-i', '--input-list', type=str, required=True, help='Input DST List.')
    parser.add_argument('-i2', '--dbtag', type=str, default='newcdbtag', help='CDB Tag. Default: newcdbtag')
    parser.add_argument('-o', '--output-dir', type=str, default='scratch/test', help='Project Directory. Default: scratch/test')
    parser.add_argument('-o2', '--job-output-dir', type=str, default=None, help='Alternate Output Directory for job output files. If provided, a symlink will be created in the main output_dir.')
    parser.add_argument('-n', '--events', type=int, default=0, help='Number of events to analyze. Default: All.')
    parser.add_argument('-n2', '--dst-per-job', type=int, default=4, help='Number of DSTs to analyze per job. Default: 4.')
    parser.add_argument('-s', '--memory', type=float, default=1.0, help='Memory (units of GB) to request per condor submission.')
    parser.add_argument('--retry-memory-step', type=float, default=0.5, help='Memory in GB to add each time job is evicted for exceeding memory. Default: 0.5 GB. Set to 0 to disable.')
    parser.add_argument('--retry-memory-max', type=float, default=6.0, help='Ceiling / max memory in GB for automatic memory retries. Default: 6.0 GB.')
    parser.add_argument('--retry-request-memory', type=str, default=None, help='Explicit retry_request_memory setting (e.g. "1.0GB, 2.0GB, 4.0GB"). If None, stepped automatically from memory + retry_memory_step up to retry_memory_max.')
    parser.add_argument('-m', '--max-retries', type=int, default=3, help='Max Condor job retries on failure. Default: 3.')
    parser.add_argument('-l', '--condor-log-dir', type=str, default='/tmp/anarde/dump', help='Condor Log Directory.')
    parser.add_argument('-e', '--email', type=str, default=None, help='Recipient email for job notifications.')
    parser.add_argument('--node', '--submit-node', type=str, default=None, help='Target submission node (e.g. sphnxuser01). Default: auto-detect best node.')
    parser.add_argument('-u', '--user', type=str, default='anarde', help="Target user ID for submit node ranking. Default: 'anarde'")

    parser.add_argument('-f2', '--src-dir', type=str, default='src', help='Source Files Directory. Default: src')
    parser.add_argument('-f3', '--condor-script', type=str, help='Condor Script.')
    parser.add_argument('-f4', '--common-errors', type=str, default='files/common-errors.txt', help='Common Errors.')

    return parser
