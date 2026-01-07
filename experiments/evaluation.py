
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
from tqdm.auto import tqdm

import itertools
import argparse
from active_learning.screening_ import active_learning
import logging
import glob

PARAMETERS = {'max_screen_size': [1000],
              'n_start': [64],
              'batch_size': [64, 32, 16],
              'architecture': ['gcn', 'mlp', 'gin', 'gat', 'rf'],
              'dataset': ['ALDH1', 'PKM2', 'VDR'],
              'seed': list(range(0,1)),
              'bias': ['random', 'small', 'large'],
              'acquisition': ['random', 'exploration', 'exploitation', 'dynamic', 'dynamicbald', 'similarity', 'bald', 'epig', 'pi', 'ei', 'ts']
              }


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', help='The path of the output directory', default='results')
    parser.add_argument('-acq', help="Acquisition function ('random', 'exploration', 'exploitation', 'dynamic', "
                                     "'similarity')", default='random')
    parser.add_argument('-bias', help='The level of bias ("random", "small", "large")', default='random')
    parser.add_argument('-arch', help='The neural network architecture ("gcn", "mlp")', default='mlp')
    parser.add_argument('-dataset', help='The dataset ("ALDH1", "PKM2", "VDR")', default='ALDH1')
    parser.add_argument('-retrain', help='Retrain the model every cycle', default='True')
    parser.add_argument('-batch_size', help='How many molecules we select each cycle', default=64)
    parser.add_argument('-n_start', help='How many molecules we have in our starting set (min=2)', default=64)
    parser.add_argument('-anchored', help='Anchor the weights', default='True')
    parser.add_argument('-scrambledx', help='Scramble the features', default='False')
    parser.add_argument('-scrambledx_seed', help='Seed for scrambling the features', default=1)
    parser.add_argument('-cycle_threshold', help='Seed for scrambling the features', default=-1)
    parser.add_argument('-cluster', help='Seed for scrambling the features', default=8)
    parser.add_argument('-test', help='Seed for scrambling the features', default=0)
    parser.add_argument('-sorted', help='Seed for scrambling the features', default='sorted')
    parser.add_argument('-start', help='Seed for scrambling the features', default=0)
    parser.add_argument('-feature', help='feature', default='fp+mf+um')
    parser.add_argument('-hidden', help='Seed for scrambling the features', default=512)
    parser.add_argument('-at_hidden', help='Seed for scrambling the features', default=256)
    parser.add_argument('-layer', help='Seed for scrambling the features', default='')
    parser.add_argument('-cycle', help='Seed for scrambling the features', default=0)
    parser.add_argument('-beta', help='Seed for scrambling the features', default=0)
    parser.add_argument('-lamda', help='Seed for scrambling the features', default=0.01)
    parser.add_argument('--input', help='input data to be trained (column : smiles, y)', default='./data/input.csv')
    parser.add_argument('--input_unlabel', help='input data to be evaluated (column : smiles)', default='./data/input_unlabel.csv')
    parser.add_argument('--output', help='output data with score. high score means high priority (column : smiles, score)', default='./result/output.csv')
    parser.add_argument("--assay_active_values", nargs="+", default=None, help="List of values to treat as Active labels (e.g., 1 active true yes).")
    parser.add_argument("--assay_inactive_values", nargs="+", default=None, help="List of values to treat as Inactive labels (e.g., 0 inactive false no).")
    parser.add_argument('--input_val_col', help='labeled input csv label column name', default='y')
    parser.add_argument('--input_unlabel_val_col', help='unlabeled input csv value column name', default='score')
    parser.add_argument('--input_smiles_col', help='labeled input csv smiles column name', default='smiles')
    parser.add_argument('--input_unlabel_smiles_col', help='unlabeled input csv smiles column name', default='smiles')
    parser.add_argument("--is_reverse", help='small value are better', action="store_true")
    args = parser.parse_args()
    
    rround=0
    args.dataset = args.input
    start = 0
    args.start = start
    PARAMETERS['acquisition'] = [args.acq]
    PARAMETERS['bias'] = [args.bias]
    PARAMETERS['dataset'] = [args.dataset]
    PARAMETERS['retrain'] = [eval(args.retrain)]
    PARAMETERS['architecture'] = [args.arch]
    PARAMETERS['batch_size'] = [int(args.batch_size)]
    PARAMETERS['n_start'] = [int(args.n_start)]
    PARAMETERS['n_start'] = [int(args.n_start)]
    PARAMETERS['anchored'] = [eval(args.anchored)]
    PARAMETERS['scrambledx'] = [eval(args.scrambledx)]
    PARAMETERS['scrambledx_seed'] = [int(args.scrambledx_seed)]
    PARAMETERS['cycle_threshold'] = [int(args.cycle_threshold)]
    PARAMETERS['cluster'] = [int(args.cluster)]
    PARAMETERS['sorted'] =  [args.sorted]
    PARAMETERS['start'] =  [int(args.start)]
    PARAMETERS['feature'] =  [args.feature]
    PARAMETERS['hidden'] =  [int(args.hidden)]
    PARAMETERS['at_hidden'] =  [int(args.at_hidden)]
    PARAMETERS['layer'] =  [args.layer]
    PARAMETERS['cycle'] =  [args.cycle]
    PARAMETERS['beta'] =  [args.beta]
    PARAMETERS['rround'] =  [rround]
    PARAMETERS['lmda'] =  [args.lamda]
    PARAMETERS['input'] =  [args.input]
    PARAMETERS['input_unlabel'] =  [args.input_unlabel]
    PARAMETERS['input_val_col'] =  [args.input_val_col]
    PARAMETERS['input_unlabel_val_col'] =  [args.input_unlabel_val_col]
    PARAMETERS['input_smiles_col'] =  [args.input_smiles_col]
    PARAMETERS['input_unlabel_smiles_col'] =  [args.input_unlabel_smiles_col]
    
    dir_name = f"./result"
    
    experiments = [dict(zip(PARAMETERS.keys(), v)) for v in itertools.product(*PARAMETERS.values())]


    for experiment in tqdm(experiments):
        results = active_learning(dir=dir_name, n_start=experiment['n_start'],
                                bias=experiment['bias'],
                                acquisition_method=experiment['acquisition'],
                                max_screen_size=experiment['max_screen_size'],
                                batch_size=experiment['batch_size'],
                                architecture=experiment['architecture'],
                                seed=experiment['seed'],
                                retrain=experiment['retrain'],
                                anchored=experiment['anchored'],
                                dataset=experiment['dataset'],
                                scrambledx=experiment['scrambledx'],
                                scrambledx_seed=experiment['scrambledx_seed'],
                                optimize_hyperparameters=False,
                                cycle_threshold=experiment['rround'],
                                beta = experiment['beta'],
                                start = experiment['start'],
                                feature = experiment['feature'], 
                                hidden=experiment['hidden'],
                                at_hidden=experiment['at_hidden'],
                                layer=experiment['layer'],
                                cycle_rnn=experiment['cycle'],
                                lmda=experiment['lmda'], 
                                input=args.input,
                                input_unlabel=args.input_unlabel,
                                output=args.output,
                                assay_active = args.assay_active_values,
                                assay_inactive = args.assay_inactive_values,
                                input_val_col=args.input_val_col,
                                input_unlabel_val_col=args.input_unlabel_val_col,
                                input_smiles_col=args.input_smiles_col,
                                input_unlabel_smiles_col=args.input_unlabel_smiles_col,
                                is_reverse=args.is_reverse)

    def cleanup_unimol_tools_logs():
        # 1) 열려 있는 FileHandler 닫기
        for logger_name in list(logging.Logger.manager.loggerDict.keys()) + [""]:
            logger = logging.getLogger(logger_name)
            for h in logger.handlers[:]:
                fn = getattr(h, "baseFilename", None)
                if fn and "unimol_tools_" in os.path.basename(fn):
                    logger.removeHandler(h)
                    try:
                        h.close()
                    except Exception:
                        pass

        # 2) 파일 삭제
        for fn in glob.glob("logs/unimol_tools_*.log"):
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass

    cleanup_unimol_tools_logs()
