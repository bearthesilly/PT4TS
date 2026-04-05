"""
Parse result_long_term_forecast.txt and generate noise robustness analysis.
Outputs: noise_robustness_report.txt + noise_robustness_plot.pdf
"""
import re
import os

# ============================================================
# Config: map (experiment, noise_tag) -> expected setting substrings
# ============================================================
EXPERIMENTS = {
    'Lag': {
        'noise_levels': [0.05, 0.15, 0.30, 0.50],
        'noise_tags':   ['0p05', '0p15', '0p30', '0p50'],
        'prior_model':  'PT_syn_lag',
        'vanilla_model': 'PT_forecast_v15',
    },
    'Periodicity': {
        'noise_levels': [0.00, 0.30, 0.60, 1.00],
        'noise_tags':   ['0p00', '0p30', '0p60', '1p00'],
        'prior_model':  'PT_syn_period',
        'vanilla_model': 'PT_forecast_v15',
    },
    'Trend': {
        'noise_levels': [0.10, 0.30, 0.50, 0.80],
        'noise_tags':   ['0p10', '0p30', '0p50', '0p80'],
        'prior_model':  'PT_syn_trend',
        'vanilla_model': 'PT_forecast_v15',
    },
}


def parse_results(filepath='result_long_term_forecast.txt'):
    """Parse setting lines and mse/mae lines into a dict keyed by setting."""
    results = {}
    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} not found!")
        return results

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line and not line.startswith('mse:'):
            setting = line
            # Next non-empty line should be metrics
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines):
                metric_line = lines[i].strip()
                m = re.match(r'mse:([\d.eE+-]+),\s*mae:([\d.eE+-]+)', metric_line)
                if m:
                    results[setting] = {
                        'mse': float(m.group(1)),
                        'mae': float(m.group(2)),
                    }
        i += 1
    return results


def find_result(results, model_id_substr, model_name):
    """Find the LAST matching result (most recent run)."""
    match = None
    for setting, metrics in results.items():
        if model_id_substr in setting and f'_{model_name}_' in setting:
            match = metrics
    return match


def main():
    results = parse_results()
    if not results:
        print("No results found. Run the experiments first.")
        return

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("NOISE ROBUSTNESS ANALYSIS: Prior Model vs Vanilla ST-PT")
    report_lines.append("=" * 80)

    # Store data for plotting
    plot_data = {}

    for exp_name, cfg in EXPERIMENTS.items():
        report_lines.append(f"\n{'='*60}")
        report_lines.append(f"  {exp_name}")
        report_lines.append(f"{'='*60}")

        header = f"{'Noise':>8s} | {'Prior MSE':>10s} {'Prior MAE':>10s} | {'Vanilla MSE':>10s} {'Vanilla MAE':>10s} | {'MSE Gap':>10s} {'Relative%':>10s}"
        report_lines.append(header)
        report_lines.append("-" * len(header))

        prior_mses, vanilla_mses, noise_vals = [], [], []

        for nl, tag in zip(cfg['noise_levels'], cfg['noise_tags']):
            model_id = f"noise_{tag}"
            prior  = find_result(results, model_id, cfg['prior_model'])
            vanilla = find_result(results, model_id, cfg['vanilla_model'])

            if prior and vanilla:
                gap = vanilla['mse'] - prior['mse']
                rel = gap / vanilla['mse'] * 100 if vanilla['mse'] > 0 else 0
                report_lines.append(
                    f"{nl:8.2f} | {prior['mse']:10.4f} {prior['mae']:10.4f} | "
                    f"{vanilla['mse']:10.4f} {vanilla['mae']:10.4f} | "
                    f"{gap:+10.4f} {rel:+9.1f}%"
                )
                prior_mses.append(prior['mse'])
                vanilla_mses.append(vanilla['mse'])
                noise_vals.append(nl)
            elif prior:
                report_lines.append(f"{nl:8.2f} | {prior['mse']:10.4f} {prior['mae']:10.4f} | {'MISSING':>10s} {'':>10s} |")
                noise_vals.append(nl)
                prior_mses.append(prior['mse'])
                vanilla_mses.append(None)
            elif vanilla:
                report_lines.append(f"{nl:8.2f} | {'MISSING':>10s} {'':>10s} | {vanilla['mse']:10.4f} {vanilla['mae']:10.4f} |")
                noise_vals.append(nl)
                prior_mses.append(None)
                vanilla_mses.append(vanilla['mse'])
            else:
                report_lines.append(f"{nl:8.2f} | {'MISSING':>10s} {'':>10s} | {'MISSING':>10s} {'':>10s} |")

        # Trend analysis
        if len(noise_vals) >= 2 and all(p is not None for p in prior_mses) and all(v is not None for v in vanilla_mses):
            gaps = [v - p for p, v in zip(prior_mses, vanilla_mses)]
            if gaps[-1] > gaps[0]:
                trend = "EXPANDING (prior advantage GROWS with noise - prior is robust!)"
            elif gaps[-1] < gaps[0] and gaps[-1] > 0:
                trend = "SHRINKING (prior advantage narrows with noise, but still positive)"
            elif gaps[-1] <= 0:
                trend = "INVERTED (vanilla catches up or surpasses prior at high noise)"
            else:
                trend = "STABLE"
            report_lines.append(f"\n  >> Gap trend: {trend}")
            report_lines.append(f"     Low-noise gap:  {gaps[0]:+.4f}")
            report_lines.append(f"     High-noise gap: {gaps[-1]:+.4f}")

        plot_data[exp_name] = {
            'noise': noise_vals,
            'prior': prior_mses,
            'vanilla': vanilla_mses,
        }

    # Summary
    report_lines.append(f"\n{'='*80}")
    report_lines.append("SUMMARY")
    report_lines.append(f"{'='*80}")
    report_lines.append("If 'MSE Gap' is positive: prior model wins (lower MSE).")
    report_lines.append("If gap EXPANDS with noise: prior is more noise-robust (best outcome).")
    report_lines.append("If gap SHRINKS with noise: prior advantage erodes under heavy noise.")
    report_lines.append("If gap becomes negative: vanilla is better at that noise level.")

    report_text = "\n".join(report_lines)
    print(report_text)

    with open("noise_robustness_report.txt", 'w') as f:
        f.write(report_text)
    print(f"\nReport saved to noise_robustness_report.txt")

    # ============================================================
    # Plot
    # ============================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, (exp_name, data) in enumerate(plot_data.items()):
            ax = axes[idx]
            ns = data['noise']
            pr = data['prior']
            va = data['vanilla']

            valid_prior = [(n, p) for n, p in zip(ns, pr) if p is not None]
            valid_vanilla = [(n, v) for n, v in zip(ns, va) if v is not None]

            if valid_prior:
                ax.plot([x[0] for x in valid_prior], [x[1] for x in valid_prior],
                        'o-', color='#2196F3', linewidth=2, markersize=8, label='Prior Model')
            if valid_vanilla:
                ax.plot([x[0] for x in valid_vanilla], [x[1] for x in valid_vanilla],
                        's--', color='#FF5722', linewidth=2, markersize=8, label='Vanilla ST-PT')

            ax.set_title(f'{exp_name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Noise Level', fontsize=12)
            ax.set_ylabel('Test MSE', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # Shade the gap region
            if valid_prior and valid_vanilla and len(valid_prior) == len(valid_vanilla):
                pn = [x[0] for x in valid_prior]
                pp = [x[1] for x in valid_prior]
                vv = [x[1] for x in valid_vanilla]
                ax.fill_between(pn, pp, vv, alpha=0.15, color='green',
                                label='Prior advantage')

        plt.suptitle('Noise Robustness: Prior Model vs Vanilla ST-PT', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig('noise_robustness_plot.pdf', bbox_inches='tight', dpi=150)
        print("Plot saved to noise_robustness_plot.pdf")

    except ImportError:
        print("matplotlib not available, skipping plot.")


if __name__ == "__main__":
    main()
