import json
from collections import defaultdict
import pandas as pd

if __name__ == '__main__':
    result = defaultdict(list)
    for result_file in ['fragile_cvx_cora_label3_1hop_boundary50_1699997494.json', 'fragile_cvx_citeseer_label3_1hop_boundary50_1700003696.json']:
        with open(f'./result/{result_file}') as fp:
            data = json.load(fp)

        for trial in range(5):
            for case in data[str(trial)]:
                attack_label = case['second_best_label']
                if str(attack_label) not in case['certified']:
                    print('case', case)
                    print('attack_label', attack_label)
                    continue

                certified = case['certified'][str(attack_label)]
                if certified['fragile'] and certified['single_predictions'][0] != attack_label:
                    result['target-node'].append(case['target_node'])
                    result['label'].append(case['ground_truth'])
                    result['prediction'].append(case['prediction'])
                    result['attack-label'].append(attack_label)
                    result['candidates'].append(certified['candidates'])
                    result['fragile-score'].append(certified['fragile_score'])
                    result['logit-diff-before'].append(certified['logit_diff_before'])
                    result['adv-prediction'].append(certified['single_predictions'][0])
                    result['adj-diff'].append(certified['adj_diff'])
                    result['perturbation'].append(certified['perturbations'][0])

    df = pd.DataFrame(result)
    df.to_csv('./result/failed_cases.csv', index=False)