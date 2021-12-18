from model.waveunet import waveunet_params
from MS2.train_apply import train_apply
import os

if __name__ == '__main__':
    # Get task information if lauched from qsub
    task_id = int(os.environ['SGE_TASK_ID']) if 'SGE_TASK_ID' in os.environ else 0
    task_first = int(os.environ['SGE_TASK_FIRST']) if 'SGE_TASK_FIRST' in os.environ else task_id
    task_last = int(os.environ['SGE_TASK_LAST']) if 'SGE_TASK_LAST' in os.environ else task_id
    num_tasks = task_last - task_first + 1
    task_index = task_id - task_first
    job_name = os.environ['JOB_NAME'] if 'JOB_NAME' in os.environ else 'experiment'

    train_apply(method='waveunet', model_args=waveunet_params.parse_args(), job_name=job_name,
                task_id=task_id, task_index=task_index, num_tasks=num_tasks)
