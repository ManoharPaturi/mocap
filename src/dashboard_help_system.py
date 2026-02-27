"""
Dashboard Help System Module
Provides glossary, metric definitions, joint angle formulas, and interpretation
guides for the motion capture GUI dashboard.

Features:
  - 15+ glossary terms with descriptions
  - Per-joint angle formulas with LaTeX notation
  - Metric interpretation guides (green/yellow/red thresholds)
  - 6+ equations in LaTeX format
  - HTML and plain-text export
  - Tooltip integration helpers for Tkinter
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


# ── Data Structures ──

@dataclass
class GlossaryEntry:
    """A single glossary term."""
    term: str
    short_description: str
    long_description: str = ''
    unit: str = ''
    category: str = 'general'
    formula_latex: str = ''
    interpretation: str = ''
    thresholds: Optional[Dict[str, float]] = None  # green/yellow/red ranges

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'term': self.term,
            'short_description': self.short_description,
            'long_description': self.long_description,
            'unit': self.unit,
            'category': self.category,
        }
        if self.formula_latex:
            d['formula_latex'] = self.formula_latex
        if self.interpretation:
            d['interpretation'] = self.interpretation
        if self.thresholds:
            d['thresholds'] = self.thresholds
        return d


@dataclass
class AngleFormula:
    """Formula definition for a joint angle."""
    name: str
    display_name: str
    landmarks: Tuple[str, str, str]  # (point_a, vertex, point_c)
    landmark_ids: Tuple[int, int, int]
    formula_latex: str
    description: str
    normal_range: Tuple[float, float] = (0.0, 180.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'display_name': self.display_name,
            'landmarks': self.landmarks,
            'landmark_ids': self.landmark_ids,
            'formula_latex': self.formula_latex,
            'description': self.description,
            'normal_range': self.normal_range,
        }


# ── Glossary Database ──

GLOSSARY: Dict[str, GlossaryEntry] = {
    'fps': GlossaryEntry(
        term='FPS',
        short_description='Frames Per Second',
        long_description='The number of frames processed per second. Higher is better for smooth tracking.',
        unit='frames/s',
        category='performance',
        thresholds={'green': 25.0, 'yellow': 15.0, 'red': 10.0},
        interpretation='Green (≥25): Smooth. Yellow (15-25): Acceptable. Red (<15): Laggy.'
    ),
    'reprojection_error': GlossaryEntry(
        term='Reprojection Error',
        short_description='3D-to-2D projection accuracy',
        long_description='Measures how well triangulated 3D points project back onto the 2D camera images. '
                         'Lower values indicate better calibration and triangulation quality.',
        unit='pixels',
        category='quality',
        formula_latex=r'e_{reproj} = \frac{1}{N} \sum_{i=1}^{N} \| \mathbf{x}_i - \pi(P_i, \mathbf{X}) \|_2',
        thresholds={'green': 5.0, 'yellow': 10.0, 'red': 20.0},
        interpretation='Green (<5px): Excellent. Yellow (5-10px): Acceptable. Red (>10px): Poor calibration.'
    ),
    'visibility': GlossaryEntry(
        term='Visibility',
        short_description='Landmark detection confidence',
        long_description='MediaPipe\'s per-landmark confidence score (0-1). Indicates how reliably '
                         'the landmark was detected. Used as weight in triangulation.',
        unit='0-1',
        category='quality',
        thresholds={'green': 0.8, 'yellow': 0.5, 'red': 0.3},
    ),
    'uncertainty': GlossaryEntry(
        term='Uncertainty',
        short_description='3D position error estimate',
        long_description='Combined estimate of 3D position error considering reprojection error, '
                         'inter-camera disagreement, visibility variance, and calibration quality.',
        unit='meters',
        category='quality',
        formula_latex=r'\sigma = w_r \cdot e_r + w_d \cdot d_{cam} + w_v \cdot \sigma_v^2 + w_o \cdot p_o + w_c \cdot e_c',
        thresholds={'green': 0.01, 'yellow': 0.03, 'red': 0.05},
        interpretation='Green (<10mm): Research-grade. Yellow (10-30mm): Clinical. Red (>50mm): Unreliable.'
    ),
    'inter_camera_disagreement': GlossaryEntry(
        term='Inter-Camera Disagreement',
        short_description='Max 3D position difference between camera estimates',
        long_description='For each landmark, computes monocular 3D estimate from each camera '
                         'and measures max pairwise Euclidean distance. Large disagreement '
                         'indicates calibration issues or partial occlusion.',
        unit='meters',
        category='quality',
        formula_latex=r'd_{disagree} = \max_{i,j} \| \mathbf{X}_i - \mathbf{X}_j \|_2',
        thresholds={'green': 0.05, 'yellow': 0.1, 'red': 0.2},
    ),
    'sync_spread': GlossaryEntry(
        term='Sync Spread',
        short_description='Timestamp difference between synchronized frames',
        long_description='Maximum time difference between frames in a synchronized batch. '
                         'Lower values mean better temporal alignment.',
        unit='ms',
        category='sync',
        thresholds={'green': 5.0, 'yellow': 20.0, 'red': 50.0},
    ),
    'clock_offset': GlossaryEntry(
        term='Clock Offset',
        short_description='Estimated time difference between PC clocks',
        long_description='Using Cristian\'s Algorithm, estimates the clock offset between the '
                         'master and camera PCs. Applied to timestamps before synchronization.',
        unit='ms',
        category='sync',
        formula_latex=r'\delta = t_{server} - (t_0 + \frac{RTT}{2})',
    ),
    'occlusion_state': GlossaryEntry(
        term='Occlusion State',
        short_description='Per-landmark visibility status',
        long_description='State machine tracking: VISIBLE (seen by ≥2 cameras), '
                         'PARTIAL (1 camera), PREDICTED (temporal extrapolation), '
                         'OCCLUDED (held position), LOST (dropped).',
        category='tracking',
    ),
    'triangulation': GlossaryEntry(
        term='Triangulation (DLT)',
        short_description='3D reconstruction from multi-view 2D observations',
        long_description='Direct Linear Transform algorithm that finds the 3D point '
                         'minimizing reprojection error across all camera views.',
        category='algorithm',
        formula_latex=r'\mathbf{x} = P \cdot \mathbf{X} \quad \Rightarrow \quad A \mathbf{X} = 0',
    ),
    'monocular_fallback': GlossaryEntry(
        term='Monocular Fallback',
        short_description='3D estimate from a single camera view',
        long_description='When a landmark is only visible in one camera, uses the camera '
                         'intrinsics and an assumed subject distance to back-project to 3D. '
                         'Less accurate than triangulation but prevents data loss.',
        category='algorithm',
        formula_latex=r'\mathbf{X}_{cam} = \begin{pmatrix} \frac{(u-c_x) \cdot Z}{f_x} \\ \frac{(v-c_y) \cdot Z}{f_y} \\ Z \end{pmatrix}',
    ),
    'one_euro_filter': GlossaryEntry(
        term='1-Euro Filter',
        short_description='Adaptive low-pass filter for jitter reduction',
        long_description='Speed-adaptive smoothing filter that reduces jitter during slow motion '
                         'while preserving responsiveness during fast motion. Applied to 3D '
                         'joint positions before kinematics computation.',
        category='algorithm',
        formula_latex=r'f_c = f_{c,min} + \beta \cdot |\dot{x}|',
    ),
    'joint_angle': GlossaryEntry(
        term='Joint Angle',
        short_description='Angle between two body segments at a joint',
        long_description='Computed as the angle at the vertex point between two limb segments. '
                         'Uses the 3D positions of three landmarks (proximal, joint, distal).',
        unit='degrees',
        category='kinematics',
        formula_latex=r'\theta = \arccos\left(\frac{\vec{v_1} \cdot \vec{v_2}}{|\vec{v_1}| \cdot |\vec{v_2}|}\right)',
    ),
    'angular_velocity': GlossaryEntry(
        term='Angular Velocity',
        short_description='Rate of change of joint angle',
        long_description='First derivative of joint angle with respect to time. '
                         'Indicates how fast a joint is moving.',
        unit='deg/s',
        category='kinematics',
        formula_latex=r'\omega = \frac{\Delta \theta}{\Delta t}',
    ),
    'linear_velocity': GlossaryEntry(
        term='Linear Velocity',
        short_description='Speed of a joint in 3D space',
        long_description='Euclidean norm of the velocity vector. Clamped to a physiological '
                         'maximum (default 6 m/s) to reject tracking outliers.',
        unit='m/s',
        category='kinematics',
        formula_latex=r'v = \sqrt{v_x^2 + v_y^2 + v_z^2}',
        thresholds={'green': 1.0, 'yellow': 3.0, 'red': 5.0},
    ),
    'consistency_score': GlossaryEntry(
        term='Skeleton Consistency',
        short_description='How well bone lengths match calibrated averages',
        long_description='Compares current bone segment lengths against running averages. '
                         'Score of 1.0 means perfect consistency. Low scores indicate '
                         'tracking errors or miscalibration.',
        unit='0-1',
        category='quality',
        thresholds={'green': 0.9, 'yellow': 0.7, 'red': 0.5},
    ),
    'center_of_mass': GlossaryEntry(
        term='Center of Mass',
        short_description='Whole-body center of mass estimate',
        long_description='Estimated using anthropometric segment mass fractions (Winter, 2009). '
                         'Useful for balance assessment and gait analysis.',
        unit='meters',
        category='kinematics',
    ),
}


# ── Angle Formulas ──

ANGLE_FORMULAS: Dict[str, AngleFormula] = {
    'elbow_right': AngleFormula(
        name='elbow_right',
        display_name='Right Elbow',
        landmarks=('R.Shoulder', 'R.Elbow', 'R.Wrist'),
        landmark_ids=(12, 14, 16),
        formula_latex=r'\theta_{elbow,R} = \angle(S_R, E_R, W_R)',
        description='Angle at the right elbow between upper arm and forearm.',
        normal_range=(10.0, 170.0),
    ),
    'elbow_left': AngleFormula(
        name='elbow_left',
        display_name='Left Elbow',
        landmarks=('L.Shoulder', 'L.Elbow', 'L.Wrist'),
        landmark_ids=(11, 13, 15),
        formula_latex=r'\theta_{elbow,L} = \angle(S_L, E_L, W_L)',
        description='Angle at the left elbow between upper arm and forearm.',
        normal_range=(10.0, 170.0),
    ),
    'knee_right': AngleFormula(
        name='knee_right',
        display_name='Right Knee',
        landmarks=('R.Hip', 'R.Knee', 'R.Ankle'),
        landmark_ids=(24, 26, 28),
        formula_latex=r'\theta_{knee,R} = \angle(H_R, K_R, A_R)',
        description='Angle at the right knee between thigh and lower leg.',
        normal_range=(10.0, 180.0),
    ),
    'knee_left': AngleFormula(
        name='knee_left',
        display_name='Left Knee',
        landmarks=('L.Hip', 'L.Knee', 'L.Ankle'),
        landmark_ids=(23, 25, 27),
        formula_latex=r'\theta_{knee,L} = \angle(H_L, K_L, A_L)',
        description='Angle at the left knee between thigh and lower leg.',
        normal_range=(10.0, 180.0),
    ),
    'shoulder_right': AngleFormula(
        name='shoulder_right',
        display_name='Right Shoulder',
        landmarks=('R.Hip', 'R.Shoulder', 'R.Elbow'),
        landmark_ids=(24, 12, 14),
        formula_latex=r'\theta_{shoulder,R} = \angle(H_R, S_R, E_R)',
        description='Angle at the right shoulder (arm abduction/flexion).',
        normal_range=(0.0, 180.0),
    ),
    'shoulder_left': AngleFormula(
        name='shoulder_left',
        display_name='Left Shoulder',
        landmarks=('L.Hip', 'L.Shoulder', 'L.Elbow'),
        landmark_ids=(23, 11, 13),
        formula_latex=r'\theta_{shoulder,L} = \angle(H_L, S_L, E_L)',
        description='Angle at the left shoulder (arm abduction/flexion).',
        normal_range=(0.0, 180.0),
    ),
    'hip_right': AngleFormula(
        name='hip_right',
        display_name='Right Hip',
        landmarks=('R.Shoulder', 'R.Hip', 'R.Knee'),
        landmark_ids=(12, 24, 26),
        formula_latex=r'\theta_{hip,R} = \angle(S_R, H_R, K_R)',
        description='Angle at the right hip (leg flexion/extension).',
        normal_range=(10.0, 180.0),
    ),
    'hip_left': AngleFormula(
        name='hip_left',
        display_name='Left Hip',
        landmarks=('L.Shoulder', 'L.Hip', 'L.Knee'),
        landmark_ids=(11, 23, 25),
        formula_latex=r'\theta_{hip,L} = \angle(S_L, H_L, K_L)',
        description='Angle at the left hip (leg flexion/extension).',
        normal_range=(10.0, 180.0),
    ),
}


# ── Help System API ──

class DashboardHelpSystem:
    """
    Provides help content for the motion capture dashboard.

    Usage:
        help_sys = DashboardHelpSystem()
        tooltip = help_sys.get_tooltip('reprojection_error')
        full_help = help_sys.get_full_help_html()
    """

    def __init__(self):
        self.glossary = GLOSSARY
        self.angle_formulas = ANGLE_FORMULAS

    def get_tooltip(self, metric_key: str) -> str:
        """Get a short tooltip string for a metric."""
        entry = self.glossary.get(metric_key)
        if not entry:
            return metric_key.replace('_', ' ').title()

        tooltip = f"{entry.term}: {entry.short_description}"
        if entry.unit:
            tooltip += f" ({entry.unit})"
        return tooltip

    def get_description(self, metric_key: str) -> str:
        """Get the full description for a metric."""
        entry = self.glossary.get(metric_key)
        if not entry:
            return ''
        return entry.long_description or entry.short_description

    def get_formula(self, metric_key: str) -> str:
        """Get the LaTeX formula for a metric."""
        entry = self.glossary.get(metric_key)
        if entry and entry.formula_latex:
            return entry.formula_latex
        formula = self.angle_formulas.get(metric_key)
        if formula:
            return formula.formula_latex
        return ''

    def get_thresholds(self, metric_key: str) -> Optional[Dict[str, float]]:
        """Get threshold values for color-coding a metric."""
        entry = self.glossary.get(metric_key)
        if entry:
            return entry.thresholds
        return None

    def classify_value(self, metric_key: str, value: float) -> str:
        """
        Classify a metric value as 'green', 'yellow', or 'red'.

        Returns 'unknown' if no thresholds are defined.
        """
        thresholds = self.get_thresholds(metric_key)
        if not thresholds:
            return 'unknown'

        green = thresholds.get('green', float('inf'))
        yellow = thresholds.get('yellow', float('inf'))

        # Different metrics have different ordering (higher-is-better vs lower-is-better)
        # For most quality metrics, lower is better (error metrics)
        # For FPS and visibility, higher is better

        higher_is_better = metric_key in ('fps', 'visibility', 'consistency_score')

        if higher_is_better:
            if value >= green:
                return 'green'
            elif value >= yellow:
                return 'yellow'
            else:
                return 'red'
        else:
            if value <= green:
                return 'green'
            elif value <= yellow:
                return 'yellow'
            else:
                return 'red'

    def get_angle_info(self, angle_name: str) -> Optional[Dict[str, Any]]:
        """Get full info about a joint angle."""
        formula = self.angle_formulas.get(angle_name)
        if formula:
            return formula.to_dict()
        return None

    def get_all_angles(self) -> List[Dict[str, Any]]:
        """Get info about all defined joint angles."""
        return [f.to_dict() for f in self.angle_formulas.values()]

    def get_glossary_by_category(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get glossary grouped by category."""
        categories: Dict[str, list] = {}
        for entry in self.glossary.values():
            cat = entry.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(entry.to_dict())
        return categories

    def get_full_help_text(self) -> str:
        """Generate full help content as plain text."""
        lines = []
        lines.append("=" * 60)
        lines.append("MOTION CAPTURE DASHBOARD — HELP & REFERENCE")
        lines.append("=" * 60)
        lines.append("")

        # Glossary by category
        categories = self.get_glossary_by_category()
        for cat_name, entries in sorted(categories.items()):
            lines.append(f"── {cat_name.upper()} ──")
            lines.append("")
            for entry in entries:
                lines.append(f"  {entry['term']}")
                lines.append(f"    {entry['short_description']}")
                if entry.get('long_description'):
                    lines.append(f"    {entry['long_description']}")
                if entry.get('unit'):
                    lines.append(f"    Unit: {entry['unit']}")
                if entry.get('formula_latex'):
                    lines.append(f"    Formula: {entry['formula_latex']}")
                if entry.get('interpretation'):
                    lines.append(f"    → {entry['interpretation']}")
                lines.append("")

        # Joint Angles
        lines.append("── JOINT ANGLE DEFINITIONS ──")
        lines.append("")
        for formula in self.angle_formulas.values():
            lines.append(f"  {formula.display_name}")
            lines.append(f"    Points: {formula.landmarks[0]} → {formula.landmarks[1]} → {formula.landmarks[2]}")
            lines.append(f"    IDs: {formula.landmark_ids}")
            lines.append(f"    Range: {formula.normal_range[0]}° – {formula.normal_range[1]}°")
            lines.append(f"    {formula.description}")
            lines.append("")

        return "\n".join(lines)

    def get_full_help_html(self) -> str:
        """Generate full help content as HTML."""
        html = ['<html><body style="font-family: sans-serif; padding: 15px;">']
        html.append('<h1>Motion Capture Dashboard — Help</h1>')

        # Glossary
        categories = self.get_glossary_by_category()
        for cat_name, entries in sorted(categories.items()):
            html.append(f'<h2>{cat_name.title()}</h2>')
            html.append('<dl>')
            for entry in entries:
                html.append(f'<dt><strong>{entry["term"]}</strong>')
                if entry.get('unit'):
                    html.append(f' <em>({entry["unit"]})</em>')
                html.append('</dt>')
                html.append(f'<dd>{entry["short_description"]}')
                if entry.get('long_description'):
                    html.append(f'<br/><small>{entry["long_description"]}</small>')
                if entry.get('interpretation'):
                    html.append(f'<br/><em>{entry["interpretation"]}</em>')
                html.append('</dd>')
            html.append('</dl>')

        # Joint Angles
        html.append('<h2>Joint Angle Formulas</h2>')
        html.append('<table border="1" cellpadding="5" cellspacing="0">')
        html.append('<tr><th>Angle</th><th>Landmarks</th><th>Range</th><th>Description</th></tr>')
        for formula in self.angle_formulas.values():
            html.append(f'<tr>')
            html.append(f'<td><strong>{formula.display_name}</strong></td>')
            html.append(f'<td>{" → ".join(formula.landmarks)}</td>')
            html.append(f'<td>{formula.normal_range[0]}°–{formula.normal_range[1]}°</td>')
            html.append(f'<td>{formula.description}</td>')
            html.append('</tr>')
        html.append('</table>')

        html.append('</body></html>')
        return '\n'.join(html)

    def export_latex(self) -> str:
        """Export all formulas as a LaTeX document fragment."""
        lines = [
            r'\section{Motion Capture Metrics}',
            '',
        ]

        for entry in self.glossary.values():
            if entry.formula_latex:
                lines.append(f'\\subsection{{{entry.term}}}')
                lines.append(f'{entry.short_description}')
                lines.append('')
                lines.append(r'\begin{equation}')
                lines.append(f'  {entry.formula_latex}')
                lines.append(r'\end{equation}')
                lines.append('')

        lines.append(r'\section{Joint Angle Definitions}')
        for formula in self.angle_formulas.values():
            lines.append(f'\\subsection{{{formula.display_name}}}')
            lines.append(f'{formula.description}')
            lines.append('')
            lines.append(r'\begin{equation}')
            lines.append(f'  {formula.formula_latex}')
            lines.append(r'\end{equation}')
            lines.append('')

        return '\n'.join(lines)
