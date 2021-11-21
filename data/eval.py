import os
import museval
from pathlib import Path
import simplejson

def evaluate_estimates(db, estimates_dir):
    museval.eval_mus_dir(db, estimates_dir, output_dir=estimates_dir)

def create_eval_store(estimates_dir):
    p = Path(estimates_dir)
    if not p.exists():
        return None
    store = museval.EvalStore()
    json_paths = p.glob('**/*.json')
    for json_path in json_paths:
        with open(json_path) as json_file:
            json_string = simplejson.loads(json_file.read())
        track_df = museval.json2df(json_string, json_path.stem)
        store.add_track(track_df)
    return store

def create_method_store(root_estimates_dir, method_dirs):
    store = museval.MethodStore()
    for method in method_dirs:
        method_estimates_dir = os.path.join(root_estimates_dir, method)
        store.add_evalstore(create_eval_store(method_estimates_dir), method)

    return store
