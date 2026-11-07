"""Refresh the measured README table without overwriting authored observations."""
import pandas as pd
from ml_core import ROOT, METRICS


def main():
    table = pd.read_csv(ROOT / 'results/metrics.csv')
    lines = ['| ML Model Name | ' + ' | '.join(METRICS) + ' |', '|---|' + '---:|' * 6]
    for _, row in table.iterrows():
        lines.append('| ' + row['ML Model Name'] + ' | ' + ' | '.join(f'{row[k]:.4f}' for k in METRICS) + ' |')
    path = ROOT / 'README.md'
    text = path.read_text(encoding='utf-8-sig')
    start, end = '<!-- RESULTS_START -->', '<!-- RESULTS_END -->'
    before, remaining = text.split(start, 1)
    _, after = remaining.split(end, 1)
    path.write_text(before + start + '\n\n' + '\n'.join(lines) + '\n\n' + end + after, encoding='utf-8')
    print('README metric table refreshed. Observations preserved: review them if results changed.')


if __name__ == '__main__':
    main()
