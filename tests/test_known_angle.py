import json
import math
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.kinematics_engine import KinematicsEngine


RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
TARGET_ERROR_DEG = 5.0


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def point_from_angle(theta_deg: float):
    theta_rad = math.radians(theta_deg)
    a = (1.0, 0.0, 0.0)
    b = (0.0, 0.0, 0.0)
    c = (math.cos(theta_rad), math.sin(theta_rad), 0.0)
    return a, b, c


if __name__ == '__main__':
    ensure_dir(RESULT_DIR)
    engine = KinematicsEngine()

    ground_truth = {
        'elbow_right': 90.0,
        'elbow_left': 180.0,
        'knee_right': 135.0,
        'knee_left': 180.0,
    }

    rows = {}
    for name, truth in ground_truth.items():
        a, b, c = point_from_angle(truth)
        measured = engine.compute_joint_angle(a, b, c)
        err = abs(measured - truth)
        rows[name] = {
            'ground_truth_deg': truth,
            'measured_deg': measured,
            'error_deg': err,
            'pass': err <= TARGET_ERROR_DEG,
        }

    json_path = os.path.join(RESULT_DIR, 'known_angle_results.json')
    md_path = os.path.join(RESULT_DIR, 'known_angle_report.md')

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'target_error_deg': TARGET_ERROR_DEG, 'results': rows}, f, indent=2)

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# Known Angle Test Report\n\n')
        f.write(f'- Target absolute error: <= {TARGET_ERROR_DEG:.2f} deg\n\n')
        for name, item in rows.items():
            f.write(
                f"- {name}: truth={item['ground_truth_deg']:.2f}, "
                f"measured={item['measured_deg']:.2f}, "
                f"error={item['error_deg']:.2f} deg | "
                f"{'PASS' if item['pass'] else 'FAIL'}\n"
            )

    print(f'Known angle test complete. JSON: {json_path}')
    print(f'Known angle test complete. Report: {md_path}')
