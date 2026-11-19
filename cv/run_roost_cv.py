"""Fine-tune one MP-only Roost source state on a CV training csv (release xfer-mp recipe), then predict the fold's test csv."""
import sys, os, random, runpy, time
variant, fold, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sys.path.insert(0, '/data/czx/GapScope_Revision_20261113/roost')
from roostlib import *
import numpy as np, torch, pandas as pd
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
CV = '/data/czx/GapScope_Revision_20261113/cv'
tr = f'{CV}/roost_in/{variant}_f{fold}_train.csv'; te = f'{CV}/roost_in/{variant}_f{fold}_test.csv'
wd = f'{CV}/roost_runs/{variant}_f{fold}_s{seed}'; os.makedirs(wd, exist_ok=True)
entry = runpy.run_path(S + '/examples/roost-example.py', run_name='archived_interface')
sys.argv = ['roost-example.py', '--train', '--robust', '--data-path', tr, '--data-id', 'cv',
            '--run-id', str(seed), '--fine-tune', f'{W}/original_comparison/models/original_mp/checkpoint-r{seed}.pth.tar',
            '--test-size', '0.0', '--val-size', '0.2', '--data-seed', '0', '--epoch', '250', '--sample', '1', '--fea-path', EMB]
args = entry['input_parser']()
assert args.epochs == 250 and args.optim == 'AdamW' and args.loss == 'L1' and args.robust and args.batch_size == 128
assert args.learning_rate == 3e-4 and args.weight_decay == 1e-6
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
os.chdir(wd); t = time.time(); entry['main'](**vars(args))
ck = f'{wd}/models/cv_s-0_t-1/checkpoint-r{seed}.pth.tar'
ids, mu, ale = predict(ck, te)
pd.DataFrame(dict(id=ids, mu=mu, ale=ale, seed=seed, fold=fold, variant=variant)).to_csv(f'{CV}/roost_runs/pred_{variant}_f{fold}_s{seed}.csv', index=False)
print('done', variant, fold, seed, time.time() - t, flush=True)
