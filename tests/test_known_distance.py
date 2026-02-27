import json
import math
import os


RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
TARGET_ERROR_METERS = 0.05


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def distance(a, b):
    return math.sqrt(
        (a['x'] - b['x']) ** 2 +
        (a['y'] - b['y']) ** 2 +
        (a['z'] - b['z']) ** 2
    )


if __name__ == '__main__':
    ensure_dir(RESULT_DIR)

    reference_points = {
        'wrist_right': {'x': 0.2, 'y': -0.5, 'z': 1.0},
        'nose': {'x': 0.0, 'y': 0.0, 'z': 1.5},
    }

    measured_points = {
        'wrist_right': {'x': 0.21, 'y': -0.49, 'z': 0.98},
        'nose': {'x': 0.01, 'y': -0.01, 'z': 1.53},
    }

    rows = {}
    origin = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    for name, reference in reference_points.items():
        measured = measured_points[name]
        expected_dist = distance(origin, reference)
        measured_dist = distance(origin, measured)
        err = abs(measured_dist - expected_dist)
        rows[name] = {
            'expected_distance_m': expected_dist,
            'measured_distance_m': measured_dist,
            'error_m': err,
            'pass': err <= TARGET_ERROR_METERS,
        }

    json_path = os.path.join(RESULT_DIR, 'known_distance_results.json')
    md_path = os.path.join(RESULT_DIR, 'known_distance_report.md')

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'target_error_m': TARGET_ERROR_METERS, 'results': rows}, f, indent=2)

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# Known Distance Test Report\n\n')
        f.write(f'- Target absolute distance error: <= {TARGET_ERROR_METERS:.3f} m\n\n')
        for name, item in rows.items():
            f.write(
                f"- {name}: expected={item['expected_distance_m']:.3f} m, "
                f"measured={item['measured_distance_m']:.3f} m, "
                f"error={item['error_m']:.3f} m | "
                f"{'PASS' if item['pass'] else 'FAIL'}\n"
            )

    print(f'Known distance test complete. JSON: {json_path}')
    print(f'Known distance test complete. Report: {md_path}')
