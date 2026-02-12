import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import webbrowser
import os
from src.database import MocapDB

class Visualizer3D:
    def __init__(self, db):
        self.db = db
        
    def plot_latest_session(self):
        """Fetch latest session data and open 3D Dashboard."""
        print("Generating Dashboard...")
        
        conn = self.db._get_connection()
        c = conn.cursor()
        
        # Get Latest Session Table
        c.execute("SELECT table_name FROM sessions ORDER BY start_time DESC LIMIT 1")
        row = c.fetchone()
        if not row or not row[0]:
            print("No sessions found.")
            conn.close()
            return
            
        table_name = row[0]
        
        # Fetch ALL Data (Robust to schema changes)
        query = f"SELECT * FROM {table_name} ORDER BY timestamp ASC"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            print("Session empty.")
            return

        # --- SETUP DASHBOARD LAYOUT ---
        # Row 1: 3D Visualization (takes more space)
        # Row 2: Arm/Leg Angles
        # Row 3: Face Metrics
        fig = make_subplots(
            rows=3, cols=2,
            row_heights=[0.6, 0.2, 0.2],
            specs=[
                [{"type": "scene", "colspan": 2}, None],
                [{"type": "xy"}, {"type": "xy"}], 
                [{"type": "xy"}, {"type": "xy"}]
            ],
            subplot_titles=("3D Motion Replay", "Arm Angles (Deg)", "Leg Angles (Deg)", "Mouth Openness", "Smile Ratio")
        )

        # --- 1. 3D ANIMATION FRAMES (Row 1) ---
        frames = []
        x_data, y_data, z_data = [], [], [] # For generating frame 0 static trace
        
        for i, row_data in df.iterrows():
            try:
                pose_json = json.loads(row_data['pose_data'])
                if not pose_json:
                    # Empty frame placeholder
                    if i==0: # Ensure at least lists exist
                        x_data, y_data, z_data = [0], [0], [0]
                    frames.append(go.Frame(data=[go.Scatter3d(x=[0],y=[0],z=[0])], name=str(i)))
                    continue
                    
                # Take Person 0
                person = pose_json[0] 
                x = [lm['x'] for lm in person]
                y = [-lm['y'] for lm in person] # Invert Y
                z = [-lm['z'] for lm in person] # Invert Z
                
                if i == 0:
                    x_data, y_data, z_data = x, y, z

                frames.append(go.Frame(data=[
                    go.Scatter3d(
                        x=x, y=y, z=z,
                        mode='markers+lines',
                        marker=dict(size=4, color='cyan'),
                        line=dict(color='white', width=2)
                    )
                ], name=str(i)))
                
            except:
                frames.append(go.Frame(data=[go.Scatter3d(x=[0],y=[0],z=[0])], name=str(i)))

        # Add Initial 3D Trace
        fig.add_trace(
            go.Scatter3d(
                x=x_data, y=y_data, z=z_data,
                mode='markers+lines',
                marker=dict(size=4, color='cyan'),
                line=dict(color='white', width=2),
                name='Skeleton'
            ),
            row=1, col=1
        )

        # --- 2. METRIC GRAPHS (Static) ---
        # We plot the entire timeline so user can see trends
        
        time_axis = df.index # Frame numbers
        
        # Helper to safely plot if column exists
        def add_metric_trace(col_name, row, col, color, label):
            if col_name in df.columns:
                fig.add_trace(
                    go.Scatter(x=time_axis, y=df[col_name], mode='lines', name=label, line=dict(color=color)),
                    row=row, col=col
                )

        # Row 2, Col 1: Arms
        add_metric_trace('Angle_Elbow_L', 2, 1, '#ff0055', 'L Elbow')
        add_metric_trace('Angle_Elbow_R', 2, 1, '#00ff88', 'R Elbow')
        
        # Row 2, Col 2: Legs
        add_metric_trace('Angle_Knee_L', 2, 2, '#ff0055', 'L Knee')
        add_metric_trace('Angle_Knee_R', 2, 2, '#00ff88', 'R Knee')
        
        # Row 3, Col 1: Face (Mouth)
        add_metric_trace('Face_Mouth_Openness', 3, 1, '#00d4ff', 'Mouth Open')
        
        # Row 3, Col 2: Face (Smile)
        add_metric_trace('Face_Smile_Ratio', 3, 2, '#ffa500', 'Smile Ratio')

        # --- LAYOUT SETTINGS ---
        fig.update_layout(
            title=f"Biomechanical Dashboard: {table_name}",
            height=900, # Tall for dashboard
            scene=dict(
                xaxis=dict(range=[0, 1], autorange=False),
                yaxis=dict(range=[-1, 0], autorange=False),
                zaxis=dict(range=[-1, 1], autorange=False),
                aspectmode='cube'
            ),
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="▶ Play Motion",
                              method="animate",
                              args=[None, {"frame": {"duration": 33, "redraw": True}, "fromcurrent": True}])]
            )]
        )
        
        # Attach frames
        fig.frames = frames
        
        # Save and Open
        filename = f"dashboard_{table_name}.html"
        fig.write_html(filename)
        webbrowser.open('file://' + os.path.realpath(filename))
        print(f"Dashboard opened: {filename}")
