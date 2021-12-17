from model.waveunet import waveunet_params
from MS2.train_apply import train_apply
import os

if __name__ == '__main__':
    task_id = int(os.environ['SGE_TASK_ID'])
    task_first = int(os.environ['SGE_TASK_FIRST'])
    task_last = int(os.environ['SGE_TASK_LAST'])
    print('I am a job task with ID %d.' % task_id)

    # preset_parser.add_argument('--preset', type=str, default=None,
    #                     help='Name of hyperparameters preset ("m1", "m2" ... "m7")')
    train_apply(method='waveunet', model_args=waveunet_params.parse_args())
