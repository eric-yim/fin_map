from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import configargparse
import json
def none_or_str(a):
    if a=='None':
        return None
    return str(a)

#Import Config File
def config():
    p = configargparse.ArgParser(default_config_files=['c.conf'])

    p.add('--default_env',help='environment variables To Create Replay Buffer',type=json.loads)
    p.add('--stock_env',help='environment variables for training runs',type=json.loads)
    p.add('--model_vars',help='variables for model',type=json.loads)
    p.add('--model_dir',help='directory for saved model',type=str)
    p.add('--model_name',help='name of saved model',type = str)
    p.add('--batch_size',type=int)
    p.add('--step_size',type=int)
    p.add('--optimizer_vars',help='variables for optimizer',type=json.loads)
    p.add('--trainer_vars',help='variables for trainer',type=json.loads)
    p.add('--n_trains',type=int)
    p.add('--save_interval',type=int)
    p.add('--file_train',help='dictionary of train days',type=str)
    
    p.add('--date_str',help='date_str for model_eval',type=none_or_str)
    p.add('--contract_str',help='contract_str for model_eval',type=none_or_str)
    p.add('--ind_p',help='plotting indice for model_eval',type=none_or_str)
    c = p.parse_args()
    return c