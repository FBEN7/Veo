# Veo Dashboard Setup Guide

Complete guide to deploy and run the Veo match analysis dashboard for your academy or club.

## Quick Start (5 minutes)

### Option 1: Docker (Recommended - Easiest)

```bash
# 1. Build the Docker image
docker build -t veo-dashboard .

# 2. Run the container
docker run -p 5000:5000 -v $(pwd)/output:/app/output veo-dashboard

# 3. Open browser to http://localhost:5000
```

### Option 2: Direct Python Installation

```bash
# 1. Install dependencies
pip install -r requirements-dashboard.txt

# 2. Run the app
python app.py

# 3. Open browser to http://localhost:5000
```

## How It Works

### Pipeline → Database → Dashboard

```
Video File (MP4)
    ↓
[main.py] - Video Analysis
    ↓
SQLite Database (output/match.db)
    ↓
[app.py] - Flask API Server
    ↓
[Dashboard] - Web Interface (Browser)
```

## Workflow for Coaches

### Step 1: Analyze Match Video

```bash
python main.py data/your_match.mp4 --output output/analysis
```

**Expected output:**
- `output/match.db` - Analysis database
- Logs showing detection progress
- Processing time: ~30-60 min for 90-minute match

### Step 2: Start Dashboard

```bash
python app.py
```

Dashboard starts at: **http://localhost:5000**

### Step 3: View & Export Results

1. **Select Match** from list
2. **View Stats**: Possession, passes, shots, player performance
3. **Export PDF**: Professional report for coaching staff
4. **Print**: Take to training session

## Dashboard Features

### Match Summary
- Team possession percentage
- Shot counts and goals
- Event breakdown (passes, tackles, etc.)
- Top performers

### Possession Timeline
- See when each team controlled the ball
- Identify momentum shifts
- Spot tactical changes

### Player Statistics Table
- Track individual distances (km)
- Top speeds by player
- Pass counts
- Rating system

### Heatmaps
- Visual representation of where players were active
- See team positioning patterns
- Identify weaknesses in defense/attack

### PDF Export
- Professional match report
- Share with coaching staff
- Print for tactical discussion

## Installation Troubleshooting

### Port 5000 Already in Use

Change the port in `app.py`:

```python
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)  # Change 5000 to 8080
```

### Database Not Found

Ensure you've run analysis first:
```bash
python main.py data/match.mp4
```

Check that `output/match.db` exists:
```bash
ls -la output/match.db
```

### CORS or Connection Errors

Dashboard can't connect to backend. Try:
1. Restart Flask app: `python app.py`
2. Clear browser cache (Ctrl+Shift+Delete)
3. Check firewall allows port 5000

### ModuleNotFoundError: No module named 'flask'

Install dependencies:
```bash
pip install -r requirements-dashboard.txt
```

## File Structure

```
veo/
├── app.py                      # Flask API server
├── templates/
│   └── index.html             # Dashboard interface
├── static/
│   └── style.css              # Professional styling
├── main.py                    # Video analysis pipeline
├── src/
│   ├── database.py            # Database schema
│   ├── events.py              # Event detection
│   ├── ball_tracking.py       # Ball tracking
│   └── ...
├── requirements-dashboard.txt # Flask dependencies
├── Dockerfile                 # For Docker deployment
└── output/
    └── match.db              # Generated after analysis
```

## Database Schema

Veo stores match data in SQLite with these tables:

- **matches**: Match metadata (label, date, duration)
- **events**: Individual actions (passes, shots, etc.)
- **player_stats**: Summary stats per player
- **team_stats**: Summary stats per team
- **possessions**: Possession chains
- **duels**: Contested situations

All data is portable in a single `match.db` file.

## API Reference

If you want to integrate with other tools:

### Get Matches
```
GET /api/matches
Returns: List of all matches
```

### Get Match Summary
```
GET /api/matches/{id}/summary
Returns: Possession, team stats, events, top players
```

### Get Player Stats
```
GET /api/matches/{id}/player-stats
Returns: Individual player metrics
```

### Export PDF
```
GET /api/matches/{id}/export/pdf
Returns: Professional PDF report
```

Full API documentation: See `app.py` for all endpoints.

## Performance

### Typical Processing Times

| Duration | GPU | CPU |
|----------|-----|-----|
| 5 min    | 2-3 min | 5-10 min |
| 45 min   | 20-30 min | 60-90 min |
| 90 min   | 40-60 min | 120-180 min |

### System Requirements

- **Minimum**: 8GB RAM, 4 CPU cores
- **Recommended**: 16GB RAM, 8 CPU cores, GPU (RTX 3060 or better)
- **Storage**: ~5GB per match (including video backup)

## Tips for Coaches

### Get the Most Value

1. **Analyze competitive matches**: Friendly matches don't provide realistic data
2. **Use with film study**: Dashboard is a starting point, not a replacement for watching video
3. **Compare player stats**: Use metrics to identify fatigue, workload, performance trends
4. **Share with players**: PDFs can motivate players by showing their performance data
5. **Track improvement**: Analyze same opponent multiple times to see tactical adjustments

### Interpreting Stats

- **Distance**: Total ground covered (fatigue indicator)
- **Top Speed**: Peak velocity (sprint capability)
- **Sprints**: Number of high-intensity efforts (fitness level)
- **Passes**: Total passing attempts (possession contributor)
- **Rating**: Composite performance score

## Support

### Common Questions

**Q: Can I analyze multiple videos at once?**
A: Yes, run analysis on each sequentially. Dashboard shows all.

**Q: Can I delete or modify matches?**
A: Direct database editing not recommended. Delete `output/match.db` to start fresh.

**Q: Does it work offline?**
A: Yes! Dashboard runs locally, no internet required.

**Q: Can I export data to Excel?**
A: Download the CSV from tables (feature coming soon). Currently use PDF.

### Reporting Issues

If you encounter errors:
1. Check terminal output for error messages
2. Verify database file exists: `ls output/match.db`
3. Try restarting: Stop Flask, clear cache, restart
4. Check Python version: Requires Python 3.8+

## Next Steps

1. ✅ Install dependencies: `pip install -r requirements-dashboard.txt`
2. ✅ Run analysis: `python main.py data/match.mp4`
3. ✅ Start dashboard: `python app.py`
4. ✅ Open http://localhost:5000
5. ✅ Select match and explore

**Happy analyzing! 🎯**
