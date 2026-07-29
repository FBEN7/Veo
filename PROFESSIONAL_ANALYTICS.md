# Veo Professional Analytics Guide

Complete documentation for advanced match analytics metrics and professional dashboard.

## 📊 Advanced Metrics Overview

### Phase 1: MVP (Ready to Deploy)

**Implemented metrics:**

#### Passing Analytics
- ✅ **Pass Completion %** - (Completed passes / Total passes) × 100
- ✅ **Key Passes** - Completed passes followed by a shot within 2 seconds
- ✅ **Pass Types** - Short/Long/Forward pass counts
- ✅ **Pass Distribution** - Passes completed to each teammate

#### Tactical Metrics
- ✅ **Position-Based Scores** - Defensive (0-10), Midfield (0-10), Attacking (0-10)
- ✅ **Primary Position** - Auto-detected from average X position (Defender/Midfielder/Forward)
- ✅ **Possession Contribution** - Each player's contribution to team possession

#### Defensive Metrics
- ✅ **Tackle Success Rate** - (Successful tackles / Total tackles) × 100
- ✅ **Interceptions** - Count of ball interceptions
- ✅ **Clearances** - Defensive clearances
- ✅ **Defensive Actions** - Sum of tackles + interceptions + clearances

#### Movement & Intensity
- ✅ **Ball Carry Distance** - Total meters player carried the ball
- ✅ **Distance Covered** - Total ground distance (existing)
- ✅ **Top Speed** - Peak velocity achieved (existing)
- ✅ **Sprints** - Number of high-intensity efforts >22 km/h (existing)

#### Player Rating
- ✅ **Composite Rating (0-10)** - Weighted score combining:
  - Pass completion: 40%
  - Positional impact: 30%
  - Defensive contribution: 20%
  - Key passes: 10%

---

### Phase 2: Next Iteration (Development Required)

#### Expected Metrics
- 📊 **xG (Expected Goals)** - Probabilistic goal model based on shot location/distance
- 📊 **xA (Expected Assists)** - Expected assists from key passes
- 📊 **Shot Quality** - Classification of shot difficulty

#### Implementation Details:
- Requires shot distance + angle calculation
- Can start with simple distance-based model
- Future: Deep learning model for sophistication

**Simple xG Model:**
```python
def estimate_xg(distance_to_goal, shot_type):
    base_xg = max(0.01, 0.5 - (distance / 200))
    if shot_type == "header":
        base_xg *= 0.8
    elif shot_type == "penalty":
        base_xg = 0.75
    return min(1.0, base_xg)
```

---

### Phase 3: Professional Features (Long-term)

#### Player Identification
- 📸 **Jersey Number Recognition** - OCR from video frames
- 📸 **Player Face Recognition** - Identify players visually
- 📸 **Video Thumbnails** - Frame capture for each player

#### Advanced Analytics
- 🎯 **Pass Networks** - Visualization of passing connections
- 🎯 **Pressure Maps** - Where defensive pressure is applied
- 🎯 **Space Control** - Territorial control metrics
- 🎯 **Transition Analytics** - Attack/defense transition efficiency

#### Situational Metrics
- ⏱️ **Set-Piece Analysis** - Corners, free-kicks, throw-ins
- ⏱️ **In-Play vs Out-of-Play** - Actions split by match state
- ⏱️ **Fatigue Analysis** - Workload monitoring by time period

---

## 🔧 Implementation

### Database Schema Updates

New columns for `player_stats`:
```sql
-- Passing
pass_completion_rate REAL DEFAULT 0.0,
key_passes INTEGER DEFAULT 0,
short_passes INTEGER DEFAULT 0,
long_passes INTEGER DEFAULT 0,
forward_passes INTEGER DEFAULT 0,

-- Defense
tackles_won INTEGER DEFAULT 0,
tackle_success_rate REAL DEFAULT 0.0,
interceptions INTEGER DEFAULT 0,
clearances INTEGER DEFAULT 0,
defensive_actions INTEGER DEFAULT 0,

-- Movement
ball_carry_distance_m REAL DEFAULT 0.0,

-- Positioning
defensive_score REAL DEFAULT 0.0,
midfield_score REAL DEFAULT 0.0,
attacking_score REAL DEFAULT 0.0,
primary_position TEXT DEFAULT 'Unknown',

-- Ratings
xg REAL DEFAULT 0.0,
xa REAL DEFAULT 0.0,
overall_rating REAL DEFAULT 0.0
```

### Pipeline Integration

```python
# In main.py, after computing events and tracks:

from src.advanced_analytics import compute_advanced_analytics

# Calculate advanced metrics
analytics = compute_advanced_analytics(events, tracks)

# Insert into database
db.insert_advanced_analytics(match_id, analytics)

# Combine with existing metrics for comprehensive player stats
```

---

## 📈 Dashboard Sections

### Overview Tab
- **Possession Timeline** - Possession % over time
- **Shot Map** - Where shots were taken (pitch visualization)
- **Heatmaps** - Movement intensity for each team

### Players Tab
- **Performance Rankings** - Sortable player table with:
  - Jersey number (currently track_id, upgrade to shirt number when available)
  - Position badge (Defender/Midfielder/Forward)
  - Rating stars (0-10 scale)
  - Key stats (distance, speed, passes, pass %, key passes, shots)

### Passing Tab
- **Pass Completion Rate** - Bar chart by team
- **Pass Types Distribution** - Pie chart (short/long/forward)
- **Pass Network** - Node-link diagram showing pass connections between players

### Defense Tab
- **Tackle Success Rate** - Bar chart
- **Defensive Actions** - Interceptions, clearances, tackles
- **Defensive Line** - Average depth of defensive line over time

### Advanced Tab
- **xG / xA Comparison** - Expected goals vs actual
- **Ball Carry Distance** - Total distance by player
- **Positional Scores** - Heatmap of defensive/midfield/attacking impact
- **Player Ratings** - Overall performance scores

---

## 👤 Player Identification Strategy

### Current Implementation (MVP)
- **Track ID** - Auto-assigned during detection (1-22)
- **Team Assignment** - Based on clustering and possession logic
- **Position** - Auto-detected from average X coordinate

### Upgrade Path (Phase 2)
- **Jersey Number Recognition** - OCR on player shirts
- **Player Database** - Match jerseys to official squad lists
- **Video Frame Capture** - Store thumbnail of each player

### Future (Phase 3)
- **Face Recognition** - Identify specific players
- **Historical Tracking** - Track same player across multiple matches
- **Player Profiles** - Season stats aggregation

---

## 📊 Interpretation Guide for Coaches

### Understanding Player Rating
```
9-10: Elite performer
  - 80%+ pass completion
  - Positive positional impact
  - Strong defensive or attacking contribution

7-8: Good performance
  - 75%+ pass completion
  - Solid positioning
  - Useful contributions

5-6: Average performance
  - 60-75% pass completion
  - Neutral impact
  - Inconsistent contributions

Below 5: Struggling
  - <60% pass completion
  - Poor positioning
  - Negative impact
```

### Reading Pass Completion
- **High completion (>80%)** - Defender or safe playmaker
- **Medium (70-80%)** - Balanced midfielder
- **Lower (60-70%)** - Attacking player taking risks
- **Very low (<60%)** - Struggling to retain possession

### Defensive Metrics
- **Tackles Won %** - Higher = better defender
- **Interceptions** - Reading game well, positioning
- **Clearances** - Desperate defense, under pressure
- **Defensive Actions Total** - Work rate indicator

### Offensive Metrics
- **Key Passes** - Chance creation ability
- **xG** - Quality of shooting opportunities
- **xA** - Creating clear-cut chances
- **Forward Passes %** - Progression-oriented play

---

## 🚀 Deployment

### Using Professional Dashboard

1. **Start pipeline with analysis:**
   ```bash
   python main.py data/match.mp4 --output output/analysis
   ```

2. **Start Flask server:**
   ```bash
   python app.py
   ```

3. **Access dashboard:**
   - Go to http://localhost:5000
   - Select "Professional" mode for advanced metrics
   - Click tabs to explore different analytics

4. **Export report:**
   - Click "Export PDF" for professional report
   - Share with coaching staff

### Docker Deployment
```bash
docker build -t veo-professional .
docker run -p 5000:5000 -v $(pwd)/output:/app/output veo-professional
```

---

## 📝 Configuration

### Adjustable Thresholds

In `src/advanced_analytics.py`:

```python
# Modify these to tune metrics
PASS_DISTANCE_THRESHOLD = 2.0  # meters, for short vs long
KEY_PASS_TIME_WINDOW = 2.0     # seconds after pass for shot
SPRINT_SPEED_THRESHOLD = 22.0  # km/h for sprint detection
DEFENSIVE_ZONE_X = 35          # meters for defender positioning
ATTACKING_ZONE_X = 70          # meters for attacker positioning
```

### Rating Weights

```python
# Adjust component weights in rating calculation
weights = {
    'pass_completion': 0.40,    # Possession retention
    'positional_impact': 0.30,  # Area control
    'defensive_actions': 0.20,  # Defensive work
    'key_passes': 0.10          # Chance creation
}
```

---

## 🔍 Quality Assurance

### Metric Validation

Before deploying to customers:

1. **Spot-Check Analysis**
   - Manually review 3-5 matches
   - Compare metrics to video observation
   - Verify accuracy within acceptable range (±10%)

2. **Edge Cases**
   - Test with poor video quality
   - Test with unusual team formations
   - Test with goalkeeper-outfield player scenarios

3. **Performance Testing**
   - Benchmark processing time
   - Monitor memory usage
   - Test on minimum hardware

---

## 📞 Support & Troubleshooting

### Common Issues

**Q: Pass completion seems off**
A: Check if pass outcomes are correctly classified. Verify "success" vs "fail" logic in events.

**Q: Player position doesn't match obvious position**
A: Position is auto-detected from average X coordinate. Adjust `DEFENSIVE_ZONE_X` and `ATTACKING_ZONE_X`.

**Q: xG numbers seem too high/low**
A: Simple distance-based model is approximate. Implement proper ML model for production.

**Q: Missing players in stats**
A: Ensure all players were detected in video. Check ball possession logic for correct team assignment.

---

## 🎯 Next Steps

### Immediate (MVP Launch)
1. ✅ Commit advanced_analytics.py
2. ✅ Integrate into pipeline (main.py)
3. ✅ Deploy professional dashboard
4. ✅ Run on real match data

### Short-term (Next Sprint)
1. Validate metrics accuracy on 10 real matches
2. Implement xG/xA models properly
3. Add pass network visualization
4. Gather customer feedback

### Long-term (Enterprise)
1. Jersey number recognition (OCR)
2. Player face recognition
3. Historical player profiles
4. Season-long analytics dashboards
5. Opponent analysis reports

---

**Ready to revolutionize football analytics for your academy! ⚽📊**
