import argparse
import torch
from exp.exp_sup import Exp_All_Task as Exp_All_Task_SUP
import random
import numpy as np
import wandb
from utils.ddp import is_main_process, init_distributed_mode
import os

VERSION = os.getenv("VERSION", "none")
WANDB_USER = os.getenv("WANDB_USER", "none")
FIXED_SEED = 444

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UniTS supervised training')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='ALL_task',
                        help='task name')
    parser.add_argument('--is_training', type=int,
                        required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True,
                        default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='UniTS',
                        help='model name')

    # data loader
    parser.add_argument('--data', type=str, required=False,
                        default='All', help='dataset type')
    parser.add_argument('--dataset_name', type=str, required=True,
                        default='', help='dataset name')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT',
                        help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--task_data_config_path', type=str,
                        default='exp/all_task.yaml', help='root path of the task and data yaml file')
    parser.add_argument('--subsample_pct', type=float,
                        default=None, help='subsample percent')

    # ddp
    parser.add_argument('--local-rank', type=int, help='local rank')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--num_workers', type=int, default=0,
                        help='data loader num workers')
    parser.add_argument("--memory_check", action="store_true", default=False)  # ! not used
    parser.add_argument("--large_model", action="store_true", default=True)

    # optimization
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int,
                        default=10, help='train epochs')
    parser.add_argument("--prompt_tune_epoch", type=int, default=0)
    parser.add_argument('--warmup_epochs', type=int,
                        default=0, help='warmup epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of train input data')
    parser.add_argument('--acc_it', type=int, default=1,
                        help='acc iteration to enlarge batch size')
    parser.add_argument('--min_lr', type=float, default=None,
                        help='optimizer min learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=0.0, help='optimizer weight decay')
    parser.add_argument('--layer_decay', type=float,
                        default=None, help='optimizer layer decay')
    parser.add_argument('--des', type=str, default='test',
                        help='exp description')
    parser.add_argument('--lradj', type=str,
                        default='supervised', help='adjust learning rate')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='save location of model checkpoints')
    parser.add_argument('--pretrained_weight', type=str, default=None,
                        help='location of pretrained model checkpoints')
    parser.add_argument('--debug', type=str,
                        default='enabled', help='disabled')
    parser.add_argument('--project_name', type=str,
                        default='tsfm-multitask', help='wandb project name')

    # model settings
    parser.add_argument('--d_model', type=int, default=512,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='num of encoder layers')
    parser.add_argument("--share_embedding",
                        action="store_true", default=False)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--prompt_num", type=int, default=5)

    # task related settings
    # forecasting task
    parser.add_argument('--inverse', action='store_true',
                        help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float,
                        default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float,
                        default=1.0, help='prior anomaly ratio (%)')

    # zero-shot-forecast-new-length
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max_offset", type=int, default=0)
    parser.add_argument('--zero_shot_forecasting_new_length',
                        type=str, default=None, help='unify')

    args = parser.parse_args()
    init_distributed_mode(args)

    print('Args in experiment:')
    print(args)

    # For step size
    win_size = 5
    for step in [2, 3, 4, 5, 6]:
        # For each learning rate
        for lr in [5e-5, 1e-5, 5e-4, 1e-4, 5e-3]:
            # For each source signal (ECG, BVP)
            for source in ['ECG', 'BVP']:
                # For each fold of the dataset, train and test the model
                for k in range(5):
                    # Set seed to ensure reproducibility
                    random.seed(FIXED_SEED)
                    torch.manual_seed(FIXED_SEED)
                    np.random.seed(FIXED_SEED)

                    args.k = k  # Set the fold
                    args.learning_rate = lr  # Set the learning rate
                    task = args.task_name
                    args.source = source  # Set the source signal
                    args.confounding = False
                    args.win_size = win_size  # Set the window size
                    args.step = step  # Set the step size

                    exp_name = f'{task}-lr{lr}-fold{k}-{source}'

                    tags = [task, f"lr{lr}", source]
                    if args.pretrained_weight is not None:
                        tags.append('pretrained')
                    else:
                        tags.append('random')

                    if is_main_process():
                        wandb.init(
                            name=exp_name,
                            entity=WANDB_USER,  # use personal wandb account
                            # set the wandb project where this run will be logged
                            project=f"{args.project_name}-step{args.step}-win{args.win_size}-{VERSION}",
                            # track hyperparameters and run metadata
                            config=args,
                            mode=args.debug,
                            group=f"{task}_lr{lr}-{source}",
                            tags=tags
                        )

                    # Run the experiment
                    Exp = Exp_All_Task_SUP

                    if args.is_training:
                        for ii in range(args.itr):
                            # setting record of experiments
                            setting = '{}_{}_{}_{}_ft{}_dm{}_el{}_{}_{}'.format(
                                args.task_name,
                                args.model_id,
                                args.model,
                                args.data,
                                args.features,
                                args.d_model,
                                args.e_layers,
                                args.des, ii)

                            exp = Exp(args)  # set experiments
                            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                            exp.train(setting)
                    else:
                        ii = 0
                        setting = '{}_{}_{}_{}_ft{}_dm{}_el{}_{}_{}'.format(
                            args.task_name,
                            args.model_id,
                            args.model,
                            args.data,
                            args.features,
                            args.d_model,
                            args.e_layers,
                            args.des, ii)

                        exp = Exp(args)  # set experiments
                        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                        exp.test(setting, load_pretrain=True)
                        torch.cuda.empty_cache()

                    # Close wandb
                    if is_main_process():
                        wandb.finish()
