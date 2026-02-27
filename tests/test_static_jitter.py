import json
import os
import random
from statistics import mean, pstdev


RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
JOINT_COUNT = 33
SECONDS = 20
FPS = 30
TARGET_STD_METERS = 0.005


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def generate_static_series(frames: int):
    baseline = {
        i: {
            'x': i * 0.002,
            'y': 1.2 - i * 0.001,
            'z': 2.0 + i * 0.0005,
        }
        for i in range(JOINT_COUNT)
    }

    series = []
    for _ in range(frames):
        frame = {}
        for i in range(JOINT_COUNT):
            frame[i] = {
                'x': baseline[i]['x'] + random.gauss(0.0, 0.0025),
                'y': baseline[i]['y'] + random.gauss(0.0, 0.0025),
                'z': baseline[i]['z'] + random.gauss(0.0, 0.0025),
            }
        series.append(frame)
    return series


def compute_stats(series):
    out = {}
    for joint_id in range(JOINT_COUNT):
        xs = [f[joint_id]['x'] for f in series]
        ys = [f[joint_id]['y'] for f in series]
        zs = [f[joint_id]['z'] for f in series]
        std_x = pstdev(xs)
        std_y = pstdev(ys)
        std_z = pstdev(zs)
        std_3d = (std_x ** 2 + std_y ** 2 + std_z ** 2) ** 0.5
        out[joint_id] = {
            'std_x_m': std_x,
            'std_y_m': std_y,
            'std_z_m': std_z,
            'std_3d_m': std_3d,
            'pass': std_3d <= TARGET_STD_METERS,
        }
    return out


def write_results(stats):
    ensure_dir(RESULT_DIR)
    json_path = os.path.join(RESULT_DIR, 'static_jitter_results.json')
    md_path = os.path.join(RESULT_DIR, 'static_jitter_report.md')

    std_values = [v['std_3d_m'] for v in stats.values()]
    pass_rate = sum(1 for v in stats.values() if v['pass']) / len(stats)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'target_std_m': TARGET_STD_METERS,
                'mean_std_3d_m': mean(std_values),
                'max_std_3d_m': max(std_values),
                'pass_rate': pass_rate,
                'per_joint': stats,
            },
            f,
            indent=2,
        )

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# Static Jitter Report\n\n')
        f.write(f'- Target 3D std: {TARGET_STD_METERS:.4f} m\n')
        f.write(f'- Mean 3D std: {mean(std_values):.4f} m\n')
        f.write(f'- Max 3D std: {max(std_values):.4f} m\n')
        f.write(f'- Pass rate: {pass_rate * 100:.1f}%\n\n')
        f.write('## Per-joint\n')
        for joint_id, item in stats.items():
            f.write(
                f"- Joint {joint_id}: std_3d={item['std_3d_m']:.4f} m | "
                f"{'PASS' if item['pass'] else 'FAIL'}\n"
            )

    return json_path, md_path


if __name__ == '__main__':
    num_frames = SECONDS * FPS
    series = generate_static_series(num_frames)
    stats = compute_stats(series)
    json_path, md_path = write_results(stats)
    print(f'Static jitter test complete. JSON: {json_path}')
    print(f'Static jitter test complete. Report: {md_path}')
