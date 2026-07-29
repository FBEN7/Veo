#!/bin/bash

# Veo Dashboard Startup Script

echo "🚀 Starting Veo Match Analysis Dashboard..."
echo ""

# Check if database exists
if [ ! -f "output/match.db" ]; then
    echo "⚠️  No database found!"
    echo "You need to run video analysis first:"
    echo "  python main.py data/your_match.mp4"
    echo ""
    exit 1
fi

# Check if dependencies are installed
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing dependencies..."
    pip install -r requirements-dashboard.txt
fi

echo "✅ Starting Flask server..."
echo "📊 Dashboard available at: http://localhost:5000"
echo "🛑 Press Ctrl+C to stop"
echo ""

python app.py
