"""Compare two independent runs, excluding measured execution times."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    left, right = Path('results'), Path('results_reproduced')
    checks = []
    for name in ['memberships.csv', 'monthly_quality.csv', 'transition_events.csv',
                 'cluster_flows.csv', 'stability.csv', 'peers_december.csv',
                 'random_baselines_december.csv', 'selection_summary.csv',
                 'model_comparison_development.csv']:
        a, b = pd.read_csv(left / name), pd.read_csv(right / name)
        a = a.drop(columns=['seconds'], errors='ignore')
        b = b.drop(columns=['seconds'], errors='ignore')
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10, atol=1e-12)
        checks.append(name)
    a, b = np.load(left / 'analysis_arrays.npz'), np.load(right / 'analysis_arrays.npz')
    for key in a.files:
        np.testing.assert_allclose(a[key], b[key], rtol=1e-10, atol=1e-12)
        checks.append('analysis_arrays.npz:' + key)
    result = {'status': 'passed', 'checks': checks, 'rtol': 1e-10, 'atol': 1e-12,
              'ignored': ['wall-clock execution time', 'output directory'],
              'scope': 'Two runs in the same Python/package/OS environment. Cross-platform bit identity is not claimed.'}
    (left / 'reproduction_check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
