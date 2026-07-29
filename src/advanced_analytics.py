"""Advanced football analytics for professional match analysis.

Calculates professional metrics including:
- Pass completion rate
- Tackle success rate
- Key passes
- Ball carry distance
- Expected goals (xG) estimation
- Expected assists (xA) estimation
- Player positioning scores (defensive/midfield/attacking)
- Heatmap data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class AdvancedAnalytics:
    """Professional match analytics engine."""

    PITCH_LENGTH = 105.0  # meters
    PITCH_WIDTH = 68.0

    # Position zones (defensive, midfield, attacking)
    DEFENSIVE_X = 35  # Own half up to 35m
    ATTACKING_X = 70  # Attacking third from 70m

    def __init__(self):
        self.pass_distance_threshold = 2.0  # meters, for "short" vs "long" passes

    def calculate_pass_metrics(self, events: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
        """Calculate pass success %, key passes, pass types."""
        passes = events[events['event_type'] == 'pass'].copy()

        if passes.empty:
            return pd.DataFrame()

        # Group by player
        player_passes = []

        for (team, track_id), group in passes.groupby(['team', 'player_track_id']):
            total = len(group)
            completed = (group['outcome'] == 'success').sum()
            success_rate = (completed / total * 100) if total > 0 else 0

            # Key passes: completed pass that led to shot
            key_passes = self._count_key_passes(group, events)

            # Pass types
            short_passes = self._count_pass_type(group, 'short')
            long_passes = self._count_pass_type(group, 'long')
            forward_passes = self._count_pass_type(group, 'forward')

            player_passes.append({
                'player_track_id': track_id,
                'team': team,
                'passes_total': total,
                'passes_completed': completed,
                'pass_completion_rate': success_rate,
                'key_passes': key_passes,
                'short_passes': short_passes,
                'long_passes': long_passes,
                'forward_passes': forward_passes,
            })

        return pd.DataFrame(player_passes)

    def calculate_tackle_metrics(self, events: pd.DataFrame) -> pd.DataFrame:
        """Calculate tackle success rate and defensive actions."""
        tackles = events[events['event_type'].isin(['tackle', 'interception', 'clearance'])].copy()

        if tackles.empty:
            return pd.DataFrame()

        player_tackles = []

        for (team, track_id), group in tackles.groupby(['team', 'player_track_id']):
            tackle_events = group[group['event_type'] == 'tackle']
            interceptions = group[group['event_type'] == 'interception']
            clearances = group[group['event_type'] == 'clearance']

            tackle_success = (tackle_events['outcome'] == 'success').sum() if not tackle_events.empty else 0
            total_tackles = len(tackle_events)
            tackle_success_rate = (tackle_success / total_tackles * 100) if total_tackles > 0 else 0

            player_tackles.append({
                'player_track_id': track_id,
                'team': team,
                'tackles': total_tackles,
                'tackles_won': tackle_success,
                'tackle_success_rate': tackle_success_rate,
                'interceptions': len(interceptions),
                'clearances': len(clearances),
                'defensive_actions': total_tackles + len(interceptions) + len(clearances),
            })

        return pd.DataFrame(player_tackles)

    def calculate_ball_carry_distance(self, tracks: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        """Calculate total distance ball was carried by each player."""
        if 'x' not in tracks.columns or 'y' not in tracks.columns:
            # Use px/py if x/y not available
            if 'px' in tracks.columns and 'py' in tracks.columns:
                tracks = tracks.rename(columns={'px': 'x', 'py': 'y'})
            else:
                return pd.DataFrame()

        player_carry = []

        for track_id in tracks['track_id'].unique():
            player_track = tracks[tracks['track_id'] == track_id].sort_values('frame')

            if len(player_track) < 2:
                continue

            # Calculate distance between consecutive frames
            x_diff = player_track['x'].diff().fillna(0)
            y_diff = player_track['y'].diff().fillna(0)
            frame_distances = np.sqrt(x_diff**2 + y_diff**2)

            # Convert to meters (assuming pixel scale is ~0.1m per pixel for 105m pitch)
            total_carry_distance = frame_distances.sum()

            # Get team
            team = player_track['team'].iloc[0]

            player_carry.append({
                'player_track_id': track_id,
                'team': team,
                'ball_carry_distance_m': total_carry_distance,
            })

        return pd.DataFrame(player_carry)

    def calculate_positional_scores(self, tracks: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        """Calculate defensive, midfield, and attacking scores based on position."""
        if 'x' not in tracks.columns:
            if 'px' in tracks.columns:
                tracks = tracks.rename(columns={'px': 'x', 'py': 'y'})
            else:
                return pd.DataFrame()

        positional_scores = []

        for track_id in tracks['track_id'].unique():
            player_track = tracks[tracks['track_id'] == track_id]

            if player_track.empty:
                continue

            # Get average X position
            avg_x = player_track['x'].mean()

            # Determine position zone
            if avg_x < self.DEFENSIVE_X:
                defensive_score = 10.0
                midfield_score = 3.0
                attacking_score = 1.0
                primary_position = 'Defender'
            elif avg_x < self.ATTACKING_X:
                defensive_score = 5.0
                midfield_score = 9.0
                attacking_score = 4.0
                primary_position = 'Midfielder'
            else:
                defensive_score = 2.0
                midfield_score = 4.0
                attacking_score = 9.5
                primary_position = 'Forward'

            team = player_track['team'].iloc[0]

            positional_scores.append({
                'player_track_id': track_id,
                'team': team,
                'defensive_score': defensive_score,
                'midfield_score': midfield_score,
                'attacking_score': attacking_score,
                'primary_position': primary_position,
                'avg_position_x': avg_x,
            })

        return pd.DataFrame(positional_scores)

    def calculate_xg_xea(self, events: pd.DataFrame) -> pd.DataFrame:
        """Estimate expected goals (xG) and expected assists (xA).

        Simple model based on shot location and distance.
        More sophisticated models would use deep learning.
        """
        shots = events[events['event_type'] == 'shot'].copy()

        if shots.empty:
            return pd.DataFrame()

        player_xg = []

        for (team, track_id), group in shots.groupby(['team', 'player_track_id']):
            # xG: estimate based on shot distance from goal
            xg = 0.0
            for _, shot in group.iterrows():
                distance = self._distance_to_goal(shot.get('location_x'), shot.get('location_y'))
                # Simple model: closer = higher xG
                shot_xg = self._estimate_shot_xg(distance, shot.get('outcome'))
                xg += shot_xg

            # xA: estimate for key passes that led to shots
            key_passes = self._count_key_passes(group, events)
            xa = key_passes * 0.15  # Rough estimate: 15% per key pass

            player_xg.append({
                'player_track_id': track_id,
                'team': team,
                'xg': round(xg, 2),
                'xa': round(xa, 2),
            })

        return pd.DataFrame(player_xg)

    def generate_heatmap_data(self, tracks: pd.DataFrame, team: str) -> List[Dict]:
        """Generate heatmap data for a specific team."""
        if 'x' not in tracks.columns:
            if 'px' in tracks.columns:
                tracks = tracks.rename(columns={'px': 'x', 'py': 'y'})
            else:
                return []

        team_tracks = tracks[tracks['team'] == team]
        heatmap = []

        for _, row in team_tracks.iterrows():
            heatmap.append({
                'x': float(row['x']) if pd.notna(row['x']) else 0,
                'y': float(row['y']) if pd.notna(row['y']) else 0,
                'intensity': 1.0
            })

        return heatmap

    def calculate_combined_rating(self, pass_metrics: pd.DataFrame, tackle_metrics: pd.DataFrame,
                                  positional: pd.DataFrame, shot_count: Dict) -> pd.DataFrame:
        """Calculate comprehensive player rating (0-10 scale)."""
        ratings = []

        for track_id in pass_metrics['player_track_id'].unique():
            pass_data = pass_metrics[pass_metrics['player_track_id'] == track_id]
            tackle_data = tackle_metrics[tackle_metrics['player_track_id'] == track_id]
            pos_data = positional[positional['player_track_id'] == track_id]

            if pass_data.empty or pos_data.empty:
                continue

            team = pass_data['team'].iloc[0]

            # Component scores
            pass_score = pass_data['pass_completion_rate'].iloc[0] / 10  # 0-10
            defensive_contribution = tackle_data['defensive_actions'].iloc[0] * 0.1 if not tackle_data.empty else 0
            positional_score = pos_data['attacking_score'].iloc[0]  # 0-10
            key_pass_score = pass_data['key_passes'].iloc[0] * 0.5  # Weight key passes

            # Combined rating (0-10)
            rating = (pass_score * 0.4 + positional_score * 0.3 +
                     defensive_contribution * 0.2 + key_pass_score * 0.1)
            rating = min(10.0, max(0.0, rating))  # Clamp to 0-10

            ratings.append({
                'player_track_id': track_id,
                'team': team,
                'overall_rating': round(rating, 1),
            })

        return pd.DataFrame(ratings)

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _count_key_passes(self, group: pd.DataFrame, events: pd.DataFrame) -> int:
        """Count passes that led to shots (key passes)."""
        if group.empty:
            return 0

        key_pass_count = 0
        for _, pass_event in group.iterrows():
            # Find shots within 2 seconds after this pass
            time_threshold = pass_event.get('timestamp_s', 0) + 2.0
            subsequent_shots = events[
                (events['timestamp_s'] > pass_event.get('timestamp_s', 0)) &
                (events['timestamp_s'] <= time_threshold) &
                (events['event_type'] == 'shot') &
                (events['team'] == pass_event['team'])
            ]

            if not subsequent_shots.empty:
                key_pass_count += 1

        return key_pass_count

    def _count_pass_type(self, group: pd.DataFrame, pass_type: str) -> int:
        """Count passes by type (short/long/forward)."""
        count = 0

        for _, row in group.iterrows():
            if pd.isna(row.get('location_x')) or pd.isna(row.get('end_location_x')):
                continue

            distance = np.sqrt(
                (row['end_location_x'] - row['location_x'])**2 +
                (row['end_location_y'] - row['location_y'])**2
            )

            if pass_type == 'short' and distance < self.pass_distance_threshold:
                count += 1
            elif pass_type == 'long' and distance >= self.pass_distance_threshold:
                count += 1
            elif pass_type == 'forward' and row['end_location_x'] > row['location_x']:
                count += 1

        return count

    def _distance_to_goal(self, x: Optional[float], y: Optional[float]) -> float:
        """Calculate distance from shot location to goal center."""
        if x is None or y is None:
            return 30.0  # Default distance

        goal_x = self.PITCH_LENGTH
        goal_y = self.PITCH_WIDTH / 2
        distance = np.sqrt((goal_x - x)**2 + (goal_y - y)**2)
        return distance

    def _estimate_shot_xg(self, distance: float, outcome: str) -> float:
        """Estimate xG for a shot based on distance and outcome."""
        # Simple model: closer shots have higher xG
        base_xg = max(0.01, 0.5 - (distance / 200))  # Decreases with distance

        # Adjust for outcome
        if outcome == 'goal':
            base_xg = max(base_xg, 0.8)
        elif outcome == 'on_target':
            base_xg *= 1.2
        elif outcome == 'off_target':
            base_xg *= 0.3

        return min(1.0, base_xg)


# ============================================================================
# Main aggregation function
# ============================================================================

def compute_advanced_analytics(
    events: pd.DataFrame,
    tracks: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Compute all advanced analytics metrics.

    Args:
        events: Event DataFrame with pass/shot/tackle data
        tracks: Track DataFrame with player positions

    Returns:
        Dictionary of analytics DataFrames
    """
    analytics = AdvancedAnalytics()

    return {
        'pass_metrics': analytics.calculate_pass_metrics(events, tracks),
        'tackle_metrics': analytics.calculate_tackle_metrics(events),
        'ball_carry': analytics.calculate_ball_carry_distance(tracks, events),
        'positional': analytics.calculate_positional_scores(tracks, events),
        'xg_xa': analytics.calculate_xg_xea(events),
    }
