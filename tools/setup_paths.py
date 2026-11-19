"""One-step setup of the paths used by the scripts.

The scripts in model/, cv/ and analysis/ keep the absolute paths of the original runs, so every number in the paper can
be traced to the file it came from (the original locations are listed in tools/paths_original.json). This script makes
them run on a new machine:

1. it builds work/ next to this repository with the directory layout of the original runs, linking each code folder of
   the repository into place (work/descr -> analysis/descr, work/cv -> cv, work/cv_rep -> cv/repeated,
   work/suppdata -> data, ...) and creating empty folders for intermediate output;
2. it creates external/ with one folder for each public input (see external/README.md for what goes where);
3. it rewrites every original absolute path in model/, cv/ and analysis/ to the matching local path.

Run it once from anywhere:

    python tools/setup_paths.py

Running it again changes nothing. Pass --dry-run to see the mapping without writing.
"""
import json, os, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORK, EXT = ROOT / 'work', ROOT / 'external'
ORIG_WORK = '/data/czx/GapScope_Revision_20261113'
# folders of the original working directory that hold code now kept in the repository
LINKS = {'benchmark': 'analysis/benchmark', 'borlido': 'analysis/borlido', 'compfloor': 'analysis/compfloor',
         'descr': 'analysis/descr', 'figs_v4': 'analysis/figs_v4', 'maps2d': 'analysis/maps2d',
         'refcheck': 'analysis/refcheck', 'struct': 'analysis/struct', 'window': 'analysis/window',
         'cv': 'cv', 'cv_rep': 'cv/repeated', 'suppdata': 'data'}
# folders of the original working directory that only receive intermediate output
OUTPUT_DIRS = ['hse', 's5', 'roost', 'probe', 'polymorph', 'cv_remote']
# public inputs: original location -> folder under external/
EXTERNAL = {'/home/czx/2024/doped/data': 'snumat',
            '/home/czx/2024/spinels/data': 'mp',
            '/home/czx/spinels/data/matbench': 'matbench',
            '/home/czx/2024/BandgapFusion_RoostReproductionWork': 'roost_release',
            '/home/czx/2024/BandgapFusion_Consolidated': 'traced_data',
            '/home/czx/2024/BandgapFusion_MeanTargetControl_20261023': 'traced_data_v1'}

def link_dir(link: pathlib.Path, target: pathlib.Path):
    if link.exists() or link.is_symlink():
        return
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name == 'nt':   # Windows without symlink rights: a directory junction needs none
            subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)], check=True, capture_output=True)
        else:
            raise

def mapping():
    m = {ORIG_WORK: WORK.as_posix(),
         '/data/czx/GapScope_cv_remote/rw': (ROOT / 'model').as_posix(),
         '/data/czx/GapScope_cv_remote/cv': (WORK / 'cv_remote').as_posix(),
         '/data/czx/GapScope_cv_remote': (WORK / 'cv_remote').as_posix(),
         '/data/czx/venvs/roost152': pathlib.Path(sys.prefix).as_posix(),
         # the Python environment of the original runs -> the interpreter running this script
         '/home/czx/2024/BandgapFusion_RoostReproductionWork/.venv/bin/python': pathlib.Path(sys.executable).as_posix(),
         # SNUMAT table export kept beside the Materials Project file in the original runs
         '/home/czx/2024/spinels/data/hse_data.csv': (EXT / 'snumat' / 'hse_data.csv').as_posix()}
    m.update({k: (EXT / v).as_posix() for k, v in EXTERNAL.items()})
    return m

def main(dry=False):
    m = mapping()
    if dry:
        print(json.dumps(m, indent=2)); return
    WORK.mkdir(exist_ok=True); EXT.mkdir(exist_ok=True)
    done = WORK / '.paths_applied.json'
    if done.exists():
        print(f'paths already set ({done}); nothing to do'); return
    for name, target in LINKS.items():
        link_dir(WORK / name, ROOT / target)
    for name in OUTPUT_DIRS:
        (WORK / name).mkdir(exist_ok=True)
    for name in EXTERNAL.values():
        (EXT / name).mkdir(exist_ok=True)
    keys = sorted(m, key=len, reverse=True)
    pat = re.compile('|'.join(re.escape(k) for k in keys))   # one pass, longest key first, values never re-read
    n = 0
    for sub in ('model', 'cv', 'analysis'):
        for p in (ROOT / sub).rglob('*'):
            if p.suffix not in ('.py', '.sh') or not p.is_file() or p.is_symlink():
                continue
            t = p.read_text(encoding='utf-8')
            u = pat.sub(lambda x: m[x.group(0)], t)
            if u != t:
                p.write_text(u, encoding='utf-8', newline=''); n += 1
    done.write_text(json.dumps(m, indent=2), encoding='utf-8')
    print(f'work/ and external/ ready under {ROOT}; rewrote {n} files')
    print('Place the public inputs as described in external/README.md before running the scripts.')

if __name__ == '__main__':
    main(dry='--dry-run' in sys.argv)
