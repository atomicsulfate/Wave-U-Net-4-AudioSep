import sys
import musdb
import models.oracles as oracles
from data.eval import evaluate_estimates

if __name__ == '__main__':
    db_path = sys.argv[1]
    estimates_path = sys.argv[2]
    oracle = oracles.get_id(sys.argv[3])

    db = musdb.DB(root=db_path, is_wav=False)
    oracles.predict_db(db, oracle, estimates_path)
    evaluate_estimates(db, estimates_path)