"""Post-match PDF report generator following SPEC_rapport_pdf.md.

Generates a 6-page commercial-quality report with:
1. Cover page
2. Executive summary (KPI cards + 3 insights)
3. Field occupation (heatmaps + tilt timeline)
4. Physical data (table + histogram)
5. Physiology by period (halves comparison)
6. Methodology & data quality
"""

import io
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch
from scipy.stats import gaussian_kde

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Colors from spec
PRIMARY_HEX = "#1B4332"
ACCENT_HEX  = "#D4A017"
LIGHT_BG_HEX = "#F7F7F5"

PRIMARY     = colors.HexColor(PRIMARY_HEX)
ACCENT      = colors.HexColor(ACCENT_HEX)
LIGHT_BG    = colors.HexColor(LIGHT_BG_HEX)
WHITE       = colors.white
GREY_LIGHT  = colors.HexColor("#E0E0E0")
GREY_TEXT   = colors.HexColor("#666666")

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm


def setup_chart_style():
    """Configure matplotlib for consistent charts."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.5,
    })


def generate_insights(payload: dict) -> list[str]:
    """Generate 3 deterministic insights from data."""
    insights = []
    kpis = payload["kpis"]
    distance = kpis["distance_km"]
    
    # Territorial dominance
    if distance["team_A"] > distance["team_B"] * 1.1:
        insights.append(
            f"Team A dominated territorially ({distance['team_A']:.1f} vs {distance['team_B']:.1f} km)."
        )
    elif distance["team_B"] > distance["team_A"] * 1.1:
        insights.append(
            f"Team B was more active ({distance['team_B']:.1f} vs {distance['team_A']:.1f} km)."
        )
    else:
        insights.append(
            f"Balanced territorial distribution ({distance['team_A']:.1f} vs {distance['team_B']:.1f} km)."
        )
    
    # Sprint intensity
    sprints = kpis["sprints"]
    total = sum(sprints.values())
    if total > 100:
        insights.append(f"High intensity match with {total} sprints detected.")
    elif total > 50:
        insights.append(f"Moderate sprint activity with {total} sprints.")
    else:
        insights.append(f"Lower-intensity match with {total} sprints.")
    
    # Fatigue
    h1_a = payload["halves"]["first"]["distance_km"].get("team_A", 0)
    h2_a = payload["halves"]["second"]["distance_km"].get("team_A", 0)
    if h1_a > 0 and h2_a < h1_a * 0.85:
        insights.append("Team A showed fatigue in the second half.")
    else:
        insights.append("Consistent intensity throughout the match.")
    
    return insights[:3]


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "CoverTitle", parent=base["Title"],
            fontSize=32, textColor=WHITE, alignment=TA_CENTER,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle", parent=base["Normal"],
            fontSize=14, textColor=WHITE, alignment=TA_CENTER,
        ),
        "h1": ParagraphStyle("H1", parent=base["Heading1"],
                            fontSize=20, textColor=PRIMARY, spaceBefore=10, spaceAfter=6),
        "h2": ParagraphStyle("H2", parent=base["Heading2"],
                            fontSize=14, textColor=PRIMARY, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("Body", parent=base["Normal"], fontSize=10, leading=14),
        "small": ParagraphStyle("Small", parent=base["Normal"],
                               fontSize=8.5, textColor=GREY_TEXT),
        "kpi_number": ParagraphStyle("KPINumber", parent=base["Normal"],
                                    fontSize=28, textColor=PRIMARY,
                                    alignment=TA_CENTER, fontName="Helvetica-Bold"),
        "kpi_label": ParagraphStyle("KPILabel", parent=base["Normal"],
                                   fontSize=9, textColor=GREY_TEXT, alignment=TA_CENTER),
    }


def render_heatmap(points: list, team: str) -> Image:
    """Render heatmap on pitch."""
    setup_chart_style()
    fig, ax = plt.subplots(figsize=(5, 8), dpi=100)
    pitch = Pitch(pitch_type="opta", pitch_color=LIGHT_BG_HEX, line_color="white")
    pitch.draw(ax=ax)
    
    if points and len(points) > 1:
        pts = np.array(points)
        try:
            z = gaussian_kde(pts.T)(pts.T)
            ax.scatter(pts[:, 0], pts[:, 1], c=z, cmap="hot", s=5, alpha=0.4)
        except:
            ax.scatter(pts[:, 0], pts[:, 1], alpha=0.1, s=10, c="red")
    
    ax.set_title(f"{team} Heatmap", fontsize=12, fontweight="bold")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=5*cm, height=8*cm)


def render_tilt(payload: dict) -> Image:
    """Render tilt timeline."""
    setup_chart_style()
    fig, ax = plt.subplots(figsize=(7, 3), dpi=100)
    
    df = pd.DataFrame(payload["tilt_timeline"]).dropna(subset=["team_A_x", "team_B_x"])
    if len(df) > 0:
        ax.plot(df["minute"], df["team_A_x"], label="Team A", linewidth=2, color=PRIMARY_HEX)
        ax.plot(df["minute"], df["team_B_x"], label="Team B", linewidth=2, color=ACCENT_HEX)
        ax.axhline(52.5, color="grey", linestyle="--", alpha=0.5)
        ax.set_xlabel("Minute")
        ax.set_ylabel("Field Position (X)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    ax.set_title("Territorial Balance Over Time", fontsize=11, fontweight="bold")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=7*cm, height=3*cm)


def render_histogram(payload: dict) -> Image:
    """Render speed histogram."""
    setup_chart_style()
    fig, ax = plt.subplots(figsize=(7, 3), dpi=100)
    
    speeds = [p["top_speed_kmh"] for p in payload["physical"]]
    if speeds:
        ax.hist(speeds, bins=15, color=PRIMARY_HEX, alpha=0.7, edgecolor="black")
        ax.axvline(20, color=ACCENT_HEX, linestyle="--", linewidth=2, label="Sprint (20 km/h)")
        ax.set_xlabel("Top Speed (km/h)")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
    
    ax.set_title("Speed Distribution", fontsize=11, fontweight="bold")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=7*cm, height=3*cm)


def build_payload(tracks: pd.DataFrame, phys: pd.DataFrame, tilt: dict,
                 poss: dict, ball_rate: float, match_label: str, duration_s: float,
                 quality_report: dict[str, Any] | None = None) -> dict:
    """Assemble payload from pipeline outputs."""
    video_duration_min = duration_s / 60.0
    
    distance_km = {}
    sprints = {}
    for team in ["team_A", "team_B"]:
        tp = phys[phys.team == team]
        distance_km[team] = tp["distance_m"].sum() / 1000.0 if not tp.empty else 0.0
        sprints[team] = int(tp["n_sprints"].sum()) if not tp.empty else 0
    
    phys_sorted = phys.sort_values("distance_m", ascending=False).head(20)
    physical = [
        {
            "track_id": int(r["track_id"]),
            "team": r["team"],
            "minutes_tracked": float(r["minutes_tracked"]),
            "distance_m": float(r["distance_m"]),
            "top_speed_kmh": float(r["top_speed_kmh"]),
            "n_sprints": int(r["n_sprints"]),
        }
        for _, r in phys_sorted.iterrows()
    ]
    
    # Heatmap points
    heatmap_points = {}
    for team in ["team_A", "team_B"]:
        tt = tracks[(tracks.team == team) & (tracks.cls == "player")]
        heatmap_points[team] = tt[["x", "y"]].values.tolist()
    
    # Tilt timeline
    fps, window = 25.0, int(5 * 60 * 25)
    tilt_data = {}
    for team in ["team_A", "team_B"]:
        tt = tracks[(tracks.team == team) & (tracks.cls == "player")]
        if not tt.empty:
            x_pos = tt.groupby("frame")["x"].mean()
            x_rolled = x_pos.rolling(window=window, center=True, min_periods=1).mean()
            for frame, x_val in x_rolled.items():
                minute = frame / (fps * 60)
                if minute not in tilt_data:
                    tilt_data[minute] = {}
                tilt_data[minute][f"{team}_x"] = float(x_val)
    
    tilt_timeline = [{"minute": m, **tilt_data[m]} for m in sorted(tilt_data.keys())]
    
    # Halves
    halftime_min = min(45, video_duration_min / 2)
    halves = {
        "first": {"distance_km": {}, "sprints": {}, "duration_min": halftime_min},
        "second": {"distance_km": {}, "sprints": {}, "duration_min": max(0, video_duration_min - halftime_min)},
    }
    for period in ["first", "second"]:
        for team in ["team_A", "team_B"]:
            tp = phys[phys.team == team]
            halves[period]["distance_km"][team] = tp["distance_m"].sum() / 1000.0 if not tp.empty else 0.0
            halves[period]["sprints"][team] = int(tp["n_sprints"].sum()) if not tp.empty else 0
    
    base_quality = {
        "ball_detection_rate": float(ball_rate),
        "n_tracks": len(tracks["track_id"].unique()),
        "warnings": [],
    }
    if quality_report:
        base_quality.update(quality_report)

    return {
        "meta": {"label": match_label, "date": datetime.now().strftime("%Y-%m-%d"),
                "video_duration_min": video_duration_min, "model": "yolov8m.pt"},
        "quality": base_quality,
        "kpis": {"distance_km": distance_km, "sprints": sprints, "field_tilt": tilt,
                "possession": poss if ball_rate >= 0.3 else {"team_A": None, "team_B": None}},
        "physical": physical,
        "tilt_timeline": tilt_timeline,
        "halves": halves,
        "heatmap_points": heatmap_points,
    }


def build_pdf(tracks: pd.DataFrame, phys: pd.DataFrame, events: list[dict[str, Any]],
             best_worst: dict, tilt: dict, possession: dict, ball_rate: float,
             quality_report: dict[str, Any] | None = None,
             match_label: str = "Match", team_event_totals: dict | None = None,
             out_path: str = "output/report.pdf") -> str:
    """Generate spec-compliant PDF report."""
    
    # Get duration
    import cv2
    try:
        cap = cv2.VideoCapture("data/match.mp4")
        duration_s = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
        cap.release()
    except:
        duration_s = 3600  # fallback
    
    payload = build_payload(
        tracks,
        phys,
        tilt,
        possession,
        ball_rate,
        match_label,
        duration_s,
        quality_report=quality_report,
    )
    insights = generate_insights(payload)
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(out_path, pagesize=A4, topMargin=MARGIN, bottomMargin=MARGIN,
                          leftMargin=MARGIN, rightMargin=MARGIN, title=match_label)
    
    styles = _styles()
    story = []
    
    # PAGE 1: COVER
    story.append(Spacer(1, 3*cm))
    story.append(Paragraph("RAPPORT POST-MATCH", styles["cover_title"]))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(payload["meta"]["label"], styles["cover_subtitle"]))
    story.append(Paragraph(payload["meta"]["date"], styles["cover_subtitle"]))
    story.append(Spacer(1, 4*cm))
    
    # Pitch decorative
    setup_chart_style()
    pitch = Pitch(pitch_type="opta", pitch_color="white", line_color="lightgrey")
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    pitch.draw(ax=ax)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    story.append(Image(buf, width=6*cm, height=4*cm))
    story.append(PageBreak())
    
    # PAGE 2: EXECUTIVE SUMMARY
    story.append(Paragraph("Executive Summary", styles["h1"]))
    story.append(Spacer(1, 0.3*cm))
    
    kpis = payload["kpis"]
    
    # KPI Cards
    kpi_items = [
        ("Distance (km)", [("Team A", str(kpis["distance_km"]["team_A"])),
                          ("Team B", str(kpis["distance_km"]["team_B"]))]),
        ("Sprints", [("Team A", str(kpis["sprints"]["team_A"])),
                    ("Team B", str(kpis["sprints"]["team_B"]))]),
    ]
    
    if kpis["possession"]["team_A"] is not None:
        kpi_items.append(("Possession*", [
            ("Team A", f"{kpis['possession']['team_A']*100:.0f}%"),
            ("Team B", f"{kpis['possession']['team_B']*100:.0f}%")
        ]))
    
    for label, teams in kpi_items:
        row_data = []
        for team_name, val in teams:
            cell = Table([[Paragraph(team_name, styles["kpi_label"])],
                         [Paragraph(val, styles["kpi_number"])]],
                        style=TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
            row_data.append(cell)
        story.append(Table([row_data], colWidths=[(PAGE_W - 2*MARGIN)/2]*2,
                          style=TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")])))
        story.append(Spacer(1, 0.3*cm))
    
    # Insights
    story.append(Paragraph("Key Insights", styles["h2"]))
    for i, insight in enumerate(insights, 1):
        story.append(Paragraph(f"<b>{i}.</b> {insight}", styles["body"]))
        story.append(Spacer(1, 0.3*cm))
    
    if kpis["possession"]["team_A"] is not None:
        story.append(Paragraph("<i>* Possession is a proxy based on player closest to ball.</i>",
                              styles["small"]))
    
    story.append(PageBreak())
    
    # PAGE 3: FIELD OCCUPATION
    story.append(Paragraph("Field Occupation & Territorial Balance", styles["h1"]))
    story.append(Spacer(1, 0.3*cm))
    
    hm_a = render_heatmap(payload["heatmap_points"]["team_A"], "Team A")
    hm_b = render_heatmap(payload["heatmap_points"]["team_B"], "Team B")
    story.append(Table([[hm_a, hm_b]], style=TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")])))
    story.append(Spacer(1, 0.5*cm))
    story.append(render_tilt(payload))
    story.append(PageBreak())
    
    # PAGE 4: PHYSICAL DATA
    story.append(Paragraph("Physical Performance", styles["h1"]))
    story.append(Spacer(1, 0.3*cm))
    
    phys_data = [["Track", "Team", "Time (min)", "Distance (m)", "Max Speed", "Sprints"]]
    for p in payload["physical"][:15]:
        phys_data.append([str(p["track_id"]), p["team"].replace("team_", ""),
                         f'{p["minutes_tracked"]:.1f}', f'{p["distance_m"]:.0f}',
                         f'{p["top_speed_kmh"]:.1f}', str(p["n_sprints"])])
    
    phys_table = Table(phys_data, colWidths=[1*cm, 1.5*cm, 1.5*cm, 1.8*cm, 1.5*cm, 1*cm])
    phys_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, GREY_LIGHT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
    ]))
    story.append(phys_table)
    story.append(Spacer(1, 0.4*cm))
    
    # Honesty box (MANDATORY per spec)
    honesty_text = ("<b>Methodological Note:</b> A player can appear under multiple tracks "
                   "(tracking interruptions). Distances are minimum estimates per segment.")
    story.append(Paragraph(honesty_text, ParagraphStyle(
        "Honesty", parent=getSampleStyleSheet()["Normal"],
        fontSize=9, backColor=colors.HexColor("#fff3cd"),
        borderPadding=8, borderColor="#D4A017")))
    
    story.append(Spacer(1, 0.4*cm))
    story.append(render_histogram(payload))
    story.append(PageBreak())
    
    # PAGE 5: HALVES COMPARISON
    story.append(Paragraph("Physiology by Period", styles["h1"]))
    story.append(Spacer(1, 0.3*cm))
    
    halves = payload["halves"]
    halves_data = [
        ["Metric", "First Half", "Second Half"],
        ["Distance (km)",
         f'{halves["first"]["distance_km"]["team_A"]:.1f} | {halves["first"]["distance_km"]["team_B"]:.1f}',
         f'{halves["second"]["distance_km"]["team_A"]:.1f} | {halves["second"]["distance_km"]["team_B"]:.1f}'],
        ["Sprints",
         f'{halves["first"]["sprints"]["team_A"]} | {halves["first"]["sprints"]["team_B"]}',
         f'{halves["second"]["sprints"]["team_A"]} | {halves["second"]["sprints"]["team_B"]}'],
    ]
    
    halves_table = Table(halves_data, colWidths=[3*cm, 4*cm, 4*cm])
    halves_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, GREY_LIGHT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
    ]))
    story.append(halves_table)
    story.append(PageBreak())
    
    # PAGE 6: METHODOLOGY
    story.append(Paragraph("Methodology & Data Quality", styles["h1"]))
    story.append(Spacer(1, 0.3*cm))
    
    quality = payload["quality"]
    meta = payload["meta"]
    
    players_cov = quality.get("players_per_frame", {})
    warns = quality.get("warnings", [])
    methodology = f"""
    <b>Data Quality Metrics</b><br/>
    • Ball detection rate: {quality['ball_detection_rate']*100:.0f}%<br/>
    • Number of tracks: {quality.get('n_tracks', quality.get('persistent_tracks', 0))}<br/>
    • Players per frame (p50/p90): {players_cov.get('p50', 0)} / {players_cov.get('p90', 0)}<br/>
    • Passes per minute: {quality.get('passes_per_min', 0)}<br/>
    • Duration: {meta['video_duration_min']:.1f} minutes<br/>
    • Model: {meta['model']}<br/>
    <br/>
    <b>Analysis Pipeline</b><br/>
    Automated video analysis using YOLOv8 for detection, ByteTrack for tracking, and K-means
    for team assignment. Physical statistics computed from pitch-projected coordinates.
    Field occupation derived from player position distributions over time.<br/>
    <br/>
    <b>Limitations</b><br/>
    • Requires fixed-camera video<br/>
    • Events are probabilistic and depend on ball visibility<br/>
    • Track IDs ≠ individual players<br/>
    • Ball detection depends on visibility<br/>
    <br/>
    <i>Generated {meta['date']} — Automated video analysis</i>
    """
    
    story.append(Paragraph(methodology, styles["body"]))
    if warns:
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("<b>Automated Reliability Warnings</b>", styles["h2"]))
        for w in warns:
            story.append(Paragraph(f"• {w}", styles["body"]))
    
    doc.build(story)
    print(f"[Report] PDF → {Path(out_path).resolve()}")
    return out_path
