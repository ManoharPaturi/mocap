
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class LiveVisualizer3D:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.points_scat = None
        self.lines_plts = []
        self.initialized = False
        
        # MediaPipe Pose Connections (Simplified)
        self.CONNECTIONS = [
            # Torso
            (11, 12), (11, 23), (12, 24), (23, 24),
            # Arms
            (11, 13), (13, 15), (12, 14), (14, 16),
            # Legs
            (23, 25), (25, 27), (24, 26), (26, 28)
        ]

    def init_plot(self):
        """Initialize the Matplotlib 3D plot."""
        if self.initialized: return
        
        plt.ion() # Interactive mode
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title("Live 3D Skeleton")
        
        # Set fixed limits (Normalized Mocap Volume)
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1) # Depth (Z)
        self.ax.set_zlim(-1, 1) # Height (Y)
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Z (Depth)')
        self.ax.set_zlabel('Y (Height)')
        
        # Invert Z and Y for conventional view if needed
        # self.ax.invert_zaxis()
        
        self.initialized = True
        print("[LiveViz] 3D Plot initialized")

    def update(self, pose_3d_data):
        """Update the plot with new 3D pose data."""
        if not self.initialized:
            self.init_plot()
            
        if not pose_3d_data:
            plt.pause(0.001)
            return

        # Extract landmarks (first person)
        if 'pose_3d' in pose_3d_data and len(pose_3d_data['pose_3d']) > 0:
            pose = pose_3d_data['pose_3d']

            if isinstance(pose, dict):
                person = [pose[idx] for idx in sorted(pose.keys())]
            elif isinstance(pose, list):
                person = pose[0] if pose and isinstance(pose[0], list) else pose
            else:
                person = []

            if not person:
                plt.pause(0.001)
                return

            xs = [lm['x'] for lm in person]
            ys = [lm['y'] for lm in person]
            zs = [lm['z'] for lm in person]
            
            self.ax.clear()
            
            # Reset Limits (MPL clears them on clear())
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
            self.ax.set_zlim(-1, 1)
            
            # Scatter Points
            self.ax.scatter(xs, zs, ys, c='cyan', marker='o')
            
            # Draw Bones
            for i, j in self.CONNECTIONS:
                if i < len(person) and j < len(person):
                    self.ax.plot(
                        [xs[i], xs[j]],
                        [zs[i], zs[j]], 
                        [ys[i], ys[j]],
                        c='white'
                    )
            
            plt.draw()
            plt.pause(0.001) # Trigger even loop

    def close(self):
        if self.initialized:
            plt.close(self.fig)
            self.initialized = False
