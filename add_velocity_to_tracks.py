"""Add velocity calculations to tracks from position data."""
import pandas as pd
import numpy as np

def add_velocity_to_tracks(tracks: pd.DataFrame, fps: int = 25) -> pd.DataFrame:
    """Calculate velocity components (vx, vy) from position changes.
    
    Uses frame-to-frame position differences and smoothing to estimate velocity.
    
    Args:
        tracks: DataFrame with columns [frame, time_s, cls, px, py, ...]
        fps: Frames per second for velocity calculation
    
    Returns:
        DataFrame with added columns [vx, vy] (pixels per frame)
    """
    tracks = tracks.copy()
    
    # Calculate velocity for each class separately
    for cls in tracks['cls'].unique():
        mask = tracks['cls'] == cls
        class_tracks = tracks[mask].copy()
        
        if cls == 'ball':
            # For ball, calculate velocity from position changes
            class_tracks = class_tracks.sort_values('frame').reset_index(drop=True)
            
            # Calculate raw velocity (pixel/frame)
            class_tracks['vx'] = class_tracks['px'].diff()
            class_tracks['vy'] = class_tracks['py'].diff()
            
            # Smooth velocity with rolling median to reduce noise
            window = 3
            class_tracks['vx'] = class_tracks['vx'].rolling(window, min_periods=1, center=True).median()
            class_tracks['vy'] = class_tracks['vy'].rolling(window, min_periods=1, center=True).median()
            
            # Fill NaN from diff with 0
            class_tracks['vx'] = class_tracks['vx'].fillna(0)
            class_tracks['vy'] = class_tracks['vy'].fillna(0)
            
        else:
            # For players, calculate per track_id
            for track_id in class_tracks['track_id'].unique():
                track_mask = (class_tracks['track_id'] == track_id)
                player_track = class_tracks[track_mask].sort_values('frame')
                
                if len(player_track) > 1:
                    idx = player_track.index
                    class_tracks.loc[idx, 'vx'] = player_track['px'].diff()
                    class_tracks.loc[idx, 'vy'] = player_track['py'].diff()
                else:
                    class_tracks.loc[player_track.index, 'vx'] = 0
                    class_tracks.loc[player_track.index, 'vy'] = 0
            
            # Fill NaN with 0
            class_tracks['vx'] = class_tracks['vx'].fillna(0)
            class_tracks['vy'] = class_tracks['vy'].fillna(0)
        
        # Update original dataframe
        tracks.loc[mask, 'vx'] = class_tracks['vx'].values
        tracks.loc[mask, 'vy'] = class_tracks['vy'].values
    
    # Ensure columns exist and are float
    if 'vx' not in tracks.columns:
        tracks['vx'] = 0.0
    if 'vy' not in tracks.columns:
        tracks['vy'] = 0.0
    
    tracks['vx'] = tracks['vx'].astype(float)
    tracks['vy'] = tracks['vy'].astype(float)
    
    return tracks


if __name__ == "__main__":
    # Test on the HSV tracks file
    tracks = pd.read_parquet("/home/user/Veo/output_hsv_test/tracks_hsv.parquet")
    
    print("Before velocity calculation:")
    print(f"  Columns: {list(tracks.columns)}")
    
    tracks = add_velocity_to_tracks(tracks, fps=25)
    
    print("\nAfter velocity calculation:")
    print(f"  Columns: {list(tracks.columns)}")
    
    # Check ball velocity
    ball_tracks = tracks[tracks['cls'] == 'ball']
    print(f"\nBall velocity statistics:")
    print(f"  Mean vx: {ball_tracks['vx'].mean():.2f} px/frame")
    print(f"  Mean vy: {ball_tracks['vy'].mean():.2f} px/frame")
    print(f"  Max speed: {np.sqrt(ball_tracks['vx']**2 + ball_tracks['vy']**2).max():.2f} px/frame")
    
    # Save updated tracks
    tracks.to_parquet("/home/user/Veo/output_hsv_test/tracks_hsv_with_velocity.parquet")
    print(f"\n✓ Saved tracks with velocity to tracks_hsv_with_velocity.parquet")
