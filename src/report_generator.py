import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft
import logging

# Configure Plot Style
plt.style.use('dark_background')

class ReportGenerator:
    def __init__(self, db):
        self.db = db
        self.results_dir = os.path.join(os.getcwd(), 'results')
        os.makedirs(self.results_dir, exist_ok=True)

    def generate_report(self, session_id=None):
        """Main entry point to generate massive report suite."""
        try:
            # 1. Fetch Data
            conn = self.db._get_connection()
            c = conn.cursor()
            
            if session_id:
                 c.execute("SELECT table_name FROM sessions WHERE id = %s", (session_id,)) if self.db.db_type == 'postgres' else \
                 c.execute("SELECT table_name FROM sessions WHERE id = ?", (session_id,))
                 row = c.fetchone()
                 if not row: return
                 table_name = row[0]
            else:
                 c.execute("SELECT table_name FROM sessions ORDER BY start_time DESC LIMIT 1")
                 row = c.fetchone()
                 if not row: return
                 table_name = row[0]
            
            print(f"Generating Massive Report for: {table_name}")
            
            query = f"SELECT * FROM {table_name} ORDER BY timestamp ASC"
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # --- CRITICAL SAFETY CHECK ---
            if df.empty or len(df) < 5:
                print("Session too short for analysis (< 5 frames). Skipping.")
                return None

            # 2. Setup Directory
            session_dir = os.path.join(self.results_dir, table_name)
            os.makedirs(session_dir, exist_ok=True)
            
            # 3. Export Clean CSV
            clean_df = df.drop(columns=['pose_data', 'face_data', 'hand_data', 'derived_data', 'id'], errors='ignore')
            clean_df.to_csv(os.path.join(session_dir, 'data.csv'), index=False)
            
            # 4. Generate Visualizations (The "15+ Plots" Suite)
            
            # A. Biomechanical Metrics
            self._plot_metrics(clean_df, session_dir)
            self._plot_kinematics(clean_df, session_dir)
            self._plot_symmetry(clean_df, session_dir)
            self._plot_correlation(clean_df, session_dir)
            self._plot_hand_trajectory(df, session_dir)
            self._plot_face_trajectory(df, session_dir)
            
            # B. Node-Specific Analysis (Mass Generation)
            # Define Key Nodes to Analyze
            KEY_NODES = {
                'Nose': 0,
                'L_Wrist': 15,
                'R_Wrist': 16,
                'L_Ankle': 27,
                'R_Ankle': 28,
                'L_Hip': 23,
                'R_Hip': 24
            }
            
            for name, idx in KEY_NODES.items():
                self._analyze_body_part(df, idx, name, session_dir)
            
            print(f"Report Generated: {session_dir} (~20+ files)")
            return session_dir
            
        except Exception as e:
            print(f"Report Generation Failed: {e}")
            return None

    def _analyze_body_part(self, df, node_idx, part_name, output_dir):
        """Generates Time Series and Frequency plots for a specific body part."""
        x, y, z = [], [], []
        
        for _, row in df.iterrows():
            try:
                pose = json.loads(row['pose_data'])[0] # Person 0
                if len(pose) > node_idx:
                    lm = pose[node_idx]
                    x.append(lm['x']); y.append(lm['y']); z.append(lm['z'])
                else:
                    x.append(np.nan); y.append(np.nan); z.append(np.nan)
            except:
                x.append(np.nan); y.append(np.nan); z.append(np.nan)
        
        # Fill missing
        node_df = pd.DataFrame({'X': x, 'Y': y, 'Z': z}).ffill().fillna(0)
        
        # 1. Time Series Plot
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs[0,0].plot(node_df['X'], color='cyan'); axs[0,0].set_title(f'{part_name} X Position')
        axs[0,1].plot(node_df['Y'], color='lime'); axs[0,1].set_title(f'{part_name} Y Position')
        axs[1,0].plot(node_df['Z'], color='magenta'); axs[1,0].set_title(f'{part_name} Z Position')
        
        # Velocity Profile (Speed)
        vel = np.sqrt(np.diff(node_df['X'])**2 + np.diff(node_df['Y'])**2)
        axs[1,1].plot(vel, color='yellow'); axs[1,1].set_title(f'{part_name} Velocity (Speed)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'time_series_{part_name}.png'))
        plt.close()
        
        # 2. Frequency Analysis (FFT)
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # FFT for X, Y, Z
        for i, col in enumerate(['X', 'Y', 'Z']):
            fft_vals = fft(node_df[col].values)
            n = len(fft_vals)
            freq = np.fft.fftfreq(n)
            
            ax = axs[i//2, i%2] # 0,0 -> 0,1 -> 1,0
            ax.plot(freq[:n//2], np.abs(fft_vals)[:n//2])
            ax.set_title(f'{part_name} {col} Frequency')
            ax.set_xlabel('Frequency (Hz)')
        
        # Spectrogram (X)
        try:
            nfft = min(256, len(node_df))
            if nfft > 0:
                axs[1,1].specgram(node_df['X'], NFFT=nfft, Fs=30)
                axs[1,1].set_title(f'{part_name} X Spectrogram')
        except:
            pass
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'frequency_{part_name}.png'))
        plt.close()

    def _plot_metrics(self, df, output_dir):
        """Plot the derived biomechanical metrics (Angles, etc.)"""
        angle_cols = [c for c in df.columns if 'Angle_' in c]
        if angle_cols:
            plt.figure(figsize=(15, 10))
            for col in angle_cols:
                plt.plot(df.index, df[col], label=col, alpha=0.7)
            plt.title("Joint Angles Over Time")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'joint_angles.png'))
            plt.close()
            
            plt.figure(figsize=(15, 6))
            sns.boxplot(data=df[angle_cols])
            plt.xticks(rotation=45)
            plt.title("Joint Angle Distribution")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'joint_angles_dist.png'))
            plt.close()

    def _plot_hand_trajectory(self, df, output_dir):
        """Plot Left vs Right Hand trajectory."""
        lx, ly, rx, ry = [], [], [], []
        for _, row in df.iterrows():
            try:
                pose = json.loads(row['pose_data'])[0]
                l = pose[15]; r = pose[16]
                lx.append(l['x']); ly.append(1-l['y']) 
                rx.append(r['x']); ry.append(1-r['y'])
            except: pass
        if not lx: return
                
        plt.figure(figsize=(10, 10))
        plt.plot(lx, ly, label='Left Hand', alpha=0.6)
        plt.plot(rx, ry, label='Right Hand', alpha=0.6, color='red')
        plt.title("Hand Trajectory")
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'hand_trajectory_path.png'))
        plt.close()

    def _plot_face_trajectory(self, df, output_dir):
        """Plot coordinates of specific face landmarks."""
        chin_y = []
        for _, row in df.iterrows():
            try:
                face = json.loads(row['face_data'])[0]
                chin_y.append(face[152]['y'])
            except: pass
        if chin_y:
            plt.figure(figsize=(12, 5))
            plt.plot(chin_y, color='purple')
            plt.title("Chin Vertical Movement (Talking Detection)")
            plt.savefig(os.path.join(output_dir, 'face_chin_movement.png'))
            plt.close()

    def _plot_kinematics(self, df, output_dir):
        """Calculate and plot Velocity and Acceleration."""
        angle_cols = [c for c in df.columns if 'Angle_' in c]
        if not angle_cols: return
        vel = df[angle_cols].diff().fillna(0)
        plt.figure(figsize=(12, 8))
        sns.heatmap(vel.abs(), cmap='viridis')
        plt.title("Movement Intensity (Angular Velocity)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'kinematics_velocity_heatmap.png'))
        plt.close()

    def _plot_symmetry(self, df, output_dir):
        """Compare Left vs Right Joints."""
        pairs = [('Angle_Elbow_L', 'Angle_Elbow_R'), ('Angle_Knee_L', 'Angle_Knee_R')]
        for l, r in pairs:
            if l in df.columns and r in df.columns:
                plt.figure(figsize=(8, 8))
                plt.scatter(df[l], df[r], alpha=0.5)
                lims = [np.min([plt.xlim(), plt.ylim()]), np.max([plt.xlim(), plt.ylim()])]
                plt.plot(lims, lims, 'r--', alpha=0.75)
                plt.title(f"Symmetry: {l} vs {r}")
                plt.savefig(os.path.join(output_dir, f'symmetry_{l}_vs_{r}.png'))
                plt.close()

    def _plot_correlation(self, df, output_dir):
        """Correlation Matrix."""
        numeric_df = df.select_dtypes(include=[np.number])
        if 'timestamp' in numeric_df.columns: numeric_df = numeric_df.drop(columns=['timestamp'])
        if numeric_df.shape[1] < 2: return
        plt.figure(figsize=(14, 12))
        sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', center=0)
        plt.title("Metric Correlation Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
        plt.close()
