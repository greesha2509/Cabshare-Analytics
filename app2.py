from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px


PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = Path(__file__).resolve().parents[2]
DATA_FILENAMES = {
    "airport": "14a9c0c7-63b4-40c5-813e-6a930fef3f9e.csv",
    "bengaluru": "Bengaluru Ola.csv",
    "transit": "Transit_Hub_Ride_Data_Filtered.csv",
    "urban_fare": "UrbanCabFare.csv",
    "ncr": "ncr_ride_bookings.csv",
}
DATA_DIR = (
    PROJECT_DIR
    if all(
        (PROJECT_DIR / filename).is_file()
        or (PROJECT_DIR / f"{filename}.gz").is_file()
        for filename in DATA_FILENAMES.values()
    )
    else WORKSPACE_DIR
)
SITE_DIR = PROJECT_DIR
DATA_PATHS = {}
for dataset, filename in DATA_FILENAMES.items():
    path = DATA_DIR / filename
    DATA_PATHS[dataset] = path if path.is_file() else Path(f"{path}.gz")

INK = "#18211F"
MUTED = "#69736E"
GREEN = "#176B55"
ORANGE = "#E58A52"


def standardize_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = (
        data.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return data


def prep_booking(data: pd.DataFrame, source: str) -> pd.DataFrame:
    column_mapping = {
        "driver_ratings": "driver_rating",
        "cancelled_rides_by_customer": "cancelled_by_customer",
        "cancelled_rides_by_driver": "cancelled_by_driver",
        "reason_for_cancelling_by_customer": "customer_cancel_reason",
        "driver_cancellation_reason": "driver_cancel_reason",
        "reason_for_cancelling_by_driver": "driver_cancel_reason",
        "incomplete_rides_reason": "incomplete_ride_reason",
    }
    data = standardize_columns(data).rename(columns=column_mapping)
    data["source_dataset"] = source
    if "booking_id" in data:
        data["booking_id"] = (
            data["booking_id"].astype("string").str.strip().str.strip('"')
            .replace(["", "nan", "None", "<NA>"], pd.NA)
        )
    return data


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [path.name for path in DATA_PATHS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required CSV dataset(s) are missing beside groupproject.py: "
            + ", ".join(missing)
        )

    bengaluru = prep_booking(pd.read_csv(DATA_PATHS["bengaluru"]), "Bengaluru")
    ncr = prep_booking(pd.read_csv(DATA_PATHS["ncr"]), "NCR")
    transit = prep_booking(pd.read_csv(DATA_PATHS["transit"]), "Transit")
    airport = standardize_columns(pd.read_csv(DATA_PATHS["airport"]))
    urban = standardize_columns(pd.read_csv(DATA_PATHS["urban_fare"]))

    transit_ids = (
        set(transit["booking_id"].dropna().unique())
        if "booking_id" in transit
        else set()
    )
    for data in (bengaluru, ncr):
        data["is_transit_hub"] = (
            data["booking_id"].isin(transit_ids) if "booking_id" in data else False
        )

    bookings = pd.concat([bengaluru, ncr], ignore_index=True, sort=False)
    if "booking_id" in bookings:
        bookings["_record_completeness"] = bookings.notna().sum(axis=1)
        bookings = bookings.sort_values("_record_completeness", ascending=False)
        with_id = bookings[bookings["booking_id"].notna()].drop_duplicates(
            "booking_id", keep="first"
        )
        without_id = bookings[bookings["booking_id"].isna()]
        bookings = pd.concat([with_id, without_id], ignore_index=True)
        bookings = bookings.drop(columns="_record_completeness")

    if "date" in bookings:
        bookings["date"] = pd.to_datetime(
            bookings["date"], errors="coerce", dayfirst=True
        )
        bookings["day_of_week"] = bookings["date"].dt.day_name()
    if "time" in bookings:
        bookings["time"] = bookings["time"].astype("string").str.strip()
        bookings["hour"] = pd.to_datetime(
            bookings["time"], format="mixed", errors="coerce"
        ).dt.hour

    numeric_booking_columns = [
        "booking_value",
        "ride_distance",
        "avg_vtat",
        "avg_ctat",
        "driver_rating",
        "customer_rating",
    ]
    for column in numeric_booking_columns:
        if column in bookings:
            bookings[column] = pd.to_numeric(bookings[column], errors="coerce")

    text_columns = [
        "booking_status",
        "vehicle_type",
        "pickup_location",
        "drop_location",
        "payment_method",
        "customer_cancel_reason",
        "driver_cancel_reason",
    ]
    for column in text_columns:
        if column in bookings:
            bookings[column] = (
                bookings[column].astype("string").str.strip().replace("", pd.NA)
            )

    if {"pickup_location", "drop_location"}.issubset(bookings.columns):
        complete_route = bookings["pickup_location"].notna() & bookings[
            "drop_location"
        ].notna()
        bookings["route"] = pd.NA
        bookings.loc[complete_route, "route"] = (
            bookings.loc[complete_route, "pickup_location"].astype(str)
            + " → "
            + bookings.loc[complete_route, "drop_location"].astype(str)
        )

    if "booking_status" in bookings:
        status = bookings["booking_status"].astype("string").str.lower().str.strip()
        bookings["cancelled_target"] = np.select(
            [
                status.str.contains("cancel", na=False),
                status.str.contains("success|complete|completed", na=False),
            ],
            [1, 0],
            default=np.nan,
        )

    bookings["is_transit_hub"] = (
        bookings.get("is_transit_hub", False).fillna(False).astype(int)
    )
    for column in ["booking_value", "ride_distance", "avg_vtat", "avg_ctat"]:
        if column in bookings:
            bookings.loc[bookings[column] < 0, column] = np.nan

    for column in [
        "hour_of_day",
        "distance_km",
        "no_of_active_drivers",
        "ride_requests",
        "demand_supply_ratio",
        "trip_duration",
        "base_fare",
        "surge_multiplier",
        "final_fare",
    ]:
        if column in urban:
            urban[column] = pd.to_numeric(urban[column], errors="coerce")

    return bookings, urban, airport


def chart_html(figure, chart_id: str) -> str:
    figure.update_layout(
        height=390,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans, sans-serif", "color": INK},
        title_font={"size": 17},
        margin={"l": 18, "r": 18, "t": 58, "b": 22},
        legend_title_text="",
        autosize=True,
    )
    figure.update_xaxes(showgrid=False, zeroline=False)
    figure.update_yaxes(gridcolor="#E8EAE4", zeroline=False)
    return figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True, "displaylogo": False},
        div_id=chart_id,
    )


def section(number: str, title: str, subtitle: str, content: str) -> str:
    return (
        '<section class="report-section">'
        f'<div class="section-kicker">SECTION {number}</div>'
        f"<h2>{title}</h2><p class=\"subtitle\">{subtitle}</p>{content}</section>"
    )


def card(label: str, value: str, note: str) -> str:
    return (
        '<article class="kpi"><div class="kpi-label">'
        f"{label}</div><div class=\"kpi-value\">{value}</div>"
        f'<div class="kpi-note">{note}</div></article>'
    )


def render_demand(bookings: pd.DataFrame) -> str:
    charts = []
    if "hour" in bookings and bookings["hour"].notna().any():
        hourly = bookings.groupby("hour").size().reset_index(name="Bookings")
        fig = px.bar(
            hourly,
            x="hour",
            y="Bookings",
            title="Booking demand by hour",
            labels={"hour": "Hour of day"},
            color_discrete_sequence=[GREEN],
        )
        fig.update_xaxes(dtick=2)
        charts.append(chart_html(fig, "demand-by-hour"))
    if "day_of_week" in bookings and bookings["day_of_week"].notna().any():
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        days = bookings["day_of_week"].value_counts().reindex(order).dropna()
        fig = px.bar(
            x=days.index,
            y=days.values,
            title="Demand by day of week",
            labels={"x": "Day", "y": "Bookings"},
            color_discrete_sequence=[ORANGE],
        )
        charts.append(chart_html(fig, "demand-by-weekday"))

    for column, label, color, chart_id in [
        ("pickup_location", "Top pickup locations", GREEN, "top-pickups"),
        ("drop_location", "Top drop locations", ORANGE, "top-drops"),
    ]:
        if column in bookings:
            values = bookings[column].value_counts().head(12).sort_values()
            fig = px.bar(
                x=values.values,
                y=values.index,
                orientation="h",
                title=label,
                labels={"x": "Bookings", "y": ""},
                color_discrete_sequence=[color],
            )
            charts.append(chart_html(fig, chart_id))

    if "route" in bookings:
        routes = bookings["route"].value_counts().head(15).sort_values()
        fig = px.bar(
            x=routes.values,
            y=routes.index,
            orientation="h",
            title="Most frequently recorded routes",
            labels={"x": "Bookings", "y": ""},
            color=routes.values,
            color_continuous_scale=["#CDE4D8", GREEN],
        )
        charts.append(chart_html(fig, "top-routes"))
    return '<div class="chart-grid">' + "".join(charts) + "</div>"


def render_fares(urban: pd.DataFrame) -> str:
    charts = []
    for column, title, color, chart_id, bins in [
        ("surge_multiplier", "Surge multiplier distribution", ORANGE, "surge-dist", 28),
        ("final_fare", "Final fare distribution", GREEN, "fare-dist", 30),
    ]:
        if column in urban and urban[column].notna().any():
            fig = px.histogram(
                urban.dropna(subset=[column]),
                x=column,
                nbins=bins,
                title=title,
                color_discrete_sequence=[color],
            )
            charts.append(chart_html(fig, chart_id))

    if {"no_of_active_drivers", "ride_requests"}.issubset(urban.columns):
        scatter = urban.dropna(subset=["no_of_active_drivers", "ride_requests"])
        if len(scatter) > 2500:
            scatter = scatter.sample(2500, random_state=42)
        fig = px.scatter(
            scatter,
            x="no_of_active_drivers",
            y="ride_requests",
            color="demand_supply_ratio" if "demand_supply_ratio" in urban else None,
            size="final_fare" if "final_fare" in urban else None,
            hover_data=[
                column
                for column in ["city", "hour_of_day", "surge_multiplier"]
                if column in urban
            ],
            title="Ride requests vs active drivers",
            color_continuous_scale=["#DCEDE5", GREEN],
        )
        charts.append(chart_html(fig, "requests-vs-drivers"))

    if {"demand_supply_ratio", "surge_multiplier"}.issubset(urban.columns):
        scatter = urban.dropna(subset=["demand_supply_ratio", "surge_multiplier"])
        if len(scatter) > 2500:
            scatter = scatter.sample(2500, random_state=42)
        fig = px.scatter(
            scatter,
            x="demand_supply_ratio",
            y="surge_multiplier",
            color="traffic_level" if "traffic_level" in urban else None,
            hover_data=[
                column
                for column in ["city", "hour_of_day", "final_fare"]
                if column in urban
            ],
            title="Demand–supply ratio vs surge multiplier",
        )
        charts.append(chart_html(fig, "ratio-vs-surge"))
    return '<div class="chart-grid">' + "".join(charts) + "</div>"


def render_cancellations(bookings: pd.DataFrame) -> str:
    charts = []
    if "booking_status" in bookings and bookings["booking_status"].notna().any():
        status = bookings["booking_status"].value_counts().rename_axis("Status").reset_index(name="Bookings")
        fig = px.pie(
            status,
            names="Status",
            values="Bookings",
            hole=0.62,
            title="Booking status mix",
            color_discrete_sequence=[GREEN, ORANGE, "#8FA99C", "#D7C7A8", "#7B8580"],
        )
        fig.update_traces(textposition="inside", textinfo="percent")
        charts.append(chart_html(fig, "status-mix"))

    reasons = []
    for column, label in [
        ("customer_cancel_reason", "Customer"),
        ("driver_cancel_reason", "Driver"),
    ]:
        if column in bookings:
            values = bookings[column].dropna().astype(str).value_counts().head(7)
            if not values.empty:
                reason_data = values.rename_axis("Reason").reset_index(name="Count")
                reason_data["Cancelled by"] = label
                reasons.append(reason_data)
    if reasons:
        fig = px.bar(
            pd.concat(reasons, ignore_index=True),
            x="Count",
            y="Reason",
            color="Cancelled by",
            orientation="h",
            barmode="group",
            title="Most common recorded cancellation reasons",
        )
        charts.append(chart_html(fig, "cancel-reasons"))
    return '<div class="chart-grid">' + "".join(charts) + "</div>"


def render_route_opportunities(bookings: pd.DataFrame) -> str:
    required = {"pickup_location", "drop_location", "hour"}
    if not required.issubset(bookings.columns):
        return '<div class="note">Pickup, drop and departure-hour fields are needed for route opportunity analysis.</div>'
    data = bookings.dropna(subset=list(required))
    if data.empty:
        return '<div class="note">No complete route and hour combinations are available.</div>'

    aggregations = {"booking_count": ("hour", "size")}
    if "ride_distance" in data:
        aggregations["average_distance"] = ("ride_distance", "mean")
    if "booking_value" in data:
        aggregations["average_fare"] = ("booking_value", "mean")
    if "is_transit_hub" in data:
        aggregations["transit_bookings"] = ("is_transit_hub", "sum")
    demand = (
        data.groupby(["pickup_location", "drop_location", "hour"])
        .agg(**aggregations)
        .reset_index()
    )
    demand["Route"] = demand["pickup_location"].astype(str) + " → " + demand[
        "drop_location"
    ].astype(str)
    threshold = demand["booking_count"].quantile(0.75)
    top = demand.nlargest(20, "booking_count")
    fig = px.bar(
        top.sort_values("booking_count"),
        x="booking_count",
        y="Route",
        color="hour",
        orientation="h",
        title="Top route–hour demand combinations",
        labels={"booking_count": "Bookings", "hour": "Departure hour"},
    )
    chart = chart_html(fig, "route-hour-demand")
    table_data = demand[demand["booking_count"] >= threshold].sort_values(
        "booking_count", ascending=False
    ).head(30)
    table = table_data.rename(
        columns={
            "pickup_location": "Pickup",
            "drop_location": "Drop",
            "hour": "Hour",
            "booking_count": "Bookings",
            "average_distance": "Average distance",
            "average_fare": "Average fare",
            "transit_bookings": "Transit bookings",
        }
    ).to_html(index=False, classes="data-table", border=0, float_format=lambda x: f"{x:.1f}")
    insight = (
        '<div class="insight"><b>Opportunity signal:</b> Route-hour combinations '
        f"at or above the 75th percentile booking count (threshold: "
        f"<b>{threshold:.0f} bookings</b>) are candidates for shared-ride testing. "
        "Repeated demand does not guarantee passenger willingness or a successful match."
        "</div>"
    )
    return chart + insight + table


def render_airport(airport: pd.DataFrame) -> str:
    if "year" not in airport:
        return '<div class="note">Airport passenger analysis requires a year field.</div>'
    metrics = [column for column in airport.columns if "passenger" in column]
    if not metrics:
        return '<div class="note">No passenger-related fields were found in the airport dataset.</div>'

    data = airport.copy()
    data["year"] = data["year"].astype("string").str.strip()
    data["_year_order"] = pd.to_numeric(
        data["year"].str.extract(r"^(\d{4})", expand=False), errors="coerce"
    )
    charts = []
    for metric in metrics:
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
        trend = data.dropna(subset=["_year_order", metric]).sort_values("_year_order")
        if trend.empty:
            continue
        fig = px.line(
            trend,
            x="year",
            y=metric,
            markers=True,
            title=f"Airport trend · {metric.replace('_', ' ')}",
            color_discrete_sequence=[GREEN],
        )
        fig.update_xaxes(type="category", tickangle=-45)
        charts.append(chart_html(fig, f"airport-{re.sub('[^a-z0-9]+', '-', metric)}"))
    return '<div class="chart-grid">' + "".join(charts) + "</div>"


def build_html(bookings: pd.DataFrame, urban: pd.DataFrame, airport: pd.DataFrame) -> str:
    total = len(bookings)
    cancel_rate = (
        bookings["cancelled_target"].mean() * 100
        if "cancelled_target" in bookings
        and bookings["cancelled_target"].notna().any()
        else np.nan
    )
    avg_fare = bookings["booking_value"].mean() if "booking_value" in bookings else np.nan
    avg_distance = (
        bookings["ride_distance"].mean() if "ride_distance" in bookings else np.nan
    )
    transit_share = (
        bookings["is_transit_hub"].mean() * 100 if total else np.nan
    )

    def format_number(value: float, suffix: str = "") -> str:
        return f"{value:,.1f}{suffix}" if pd.notna(value) else "—"

    kpis = "".join(
        [
            card("Booking records", f"{total:,}", "After consolidation"),
            card("Cancellation rate", format_number(cancel_rate, "%"), "Classifiable bookings"),
            card("Average booking value", f"₹{avg_fare:,.0f}" if pd.notna(avg_fare) else "—", "Recorded booking value"),
            card("Average ride distance", format_number(avg_distance), "Dataset distance units"),
            card("Transit-linked share", format_number(transit_share, "%"), "Matched transit booking IDs"),
        ]
    )

    insights = []
    if pd.notna(cancel_rate):
        insights.append(
            f"Cancellation rate among classifiable bookings is {cancel_rate:.1f}%; "
            "prioritise reliability signals and clear cancellation policies."
        )
    if "hour" in bookings and bookings["hour"].notna().any():
        peak_hour = int(bookings["hour"].value_counts().idxmax())
        insights.append(
            f"Recorded booking volume peaks at {peak_hour:02d}:00; test proactive "
            "matching and departure suggestions around this period."
        )
    if "route" in bookings and bookings["route"].notna().any():
        top_route = str(bookings["route"].value_counts().idxmax())
        insights.append(
            f"The most frequently recorded route is {top_route}; consider it as a "
            "candidate route for a controlled cab-sharing pilot."
        )
    if total:
        insights.append(
            f"{transit_share:.1f}% of records match transit-hub booking IDs; "
            "make airport and railway pickup flows easy to discover."
        )
    insight_cards = "".join(f'<div class="insight">{text}</div>' for text in insights)

    sections = [
        section(
            "01",
            "Understand commuter demand",
            "When, where and along which routes are bookings concentrated?",
            render_demand(bookings),
        ),
        section(
            "02",
            "Make fares more predictable",
            "Explore ride demand, driver availability and surge pricing patterns.",
            render_fares(urban),
        ),
        section(
            "03",
            "Find friction in the ride experience",
            "Cancellation volume and recorded reasons can indicate where reliability breaks down.",
            render_cancellations(bookings),
        ),
        section(
            "04",
            "Spot cab-sharing opportunities",
            "Repeated route-hour demand can guide controlled matching experiments.",
            render_route_opportunities(bookings),
        ),
        section(
            "05",
            "Explore passenger matching",
            "Prioritise repeated routes and departure windows; route-hour opportunities are signals, not confirmed matches.",
            '<div class="note"><b>Matching hypothesis:</b> journeys with the same pickup and drop locations and nearby departure times are candidates for passenger matching. Validate willingness, safety, and actual match rates with a controlled pilot.</div>',
        ),
        section(
            "06",
            "Evaluate predictive models",
            "Model scores are intentionally not fabricated in a static report.",
            '<div class="note">Cancellation classification and fare prediction require training and a held-out evaluation. Run those analyses in the Colab workflow before presenting model scores; this static site does not train models in a visitor’s browser.</div>',
        ),
        section(
            "07",
            "Understand airport travel context",
            "Passenger movement trends provide broader context for airport-linked commuter demand.",
            render_airport(airport),
        ),
        section(
            "08",
            "From analysis to product direction",
            "Translate repeated patterns into testable product opportunities.",
            insight_cards
            + '<div class="note"><b>Product opportunities to validate</b><ul>'
            "<li>Smart matching for similar routes and departure windows</li>"
            "<li>Transparent fare splitting before ride confirmation</li>"
            "<li>Alternative pickup points to improve match availability</li>"
            "<li>Departure suggestions based on demand patterns</li>"
            "<li>Clear safety and trip-status information</li>"
            "</ul></div>",
        ),
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="A data-led analysis of commuter demand, fares, cancellations and cab-sharing opportunities.">
<title>CabShare | Analytics Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{{--paper:#F5F4F0;--ink:{INK};--muted:{MUTED};--green:{GREEN};--line:#E3E5DF}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:'DM Sans',sans-serif}}
main{{max-width:1440px;margin:auto;padding:32px 28px 64px}}
.hero{{background:#18211F;color:#F5F4F0;padding:40px 42px;border-radius:22px;margin-bottom:26px}}
.eyebrow,.section-kicker{{font-family:'DM Mono',monospace;font-size:.72rem;letter-spacing:.13em;text-transform:uppercase;color:var(--green)}}
.hero .eyebrow{{color:#A7D7C2}}
h1{{font-size:clamp(2.4rem,6vw,4.2rem);line-height:1.02;letter-spacing:-.05em;margin:12px 0 16px}}
.hero p{{max-width:780px;color:#C4CEC8;font-size:1.05rem;line-height:1.6}}
.source-note{{color:var(--muted);margin:18px 0 32px}}
.kpis{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:12px 0 36px}}
.kpi,.chart,.data-table,.note{{background:white;border:1px solid #E7E8E2;border-radius:16px}}
.kpi{{padding:18px;min-height:120px}}
.kpi-label{{font-size:.76rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:600}}
.kpi-value{{font-size:clamp(1.5rem,3vw,2rem);font-weight:800;letter-spacing:-.05em;margin:8px 0}}
.kpi-note{{font-size:.75rem;color:#84908A}}
.report-section{{border-top:1px solid #DADDD6;padding-top:24px;margin-top:42px}}
h2{{font-size:clamp(1.55rem,4vw,2rem);letter-spacing:-.04em;margin:8px 0}}
.subtitle{{color:var(--muted);font-size:1rem;margin:0 0 20px;line-height:1.55}}
.chart-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}
.chart-grid>div{{min-width:0;background:white;border:1px solid #E7E8E2;border-radius:14px;padding:8px;overflow:hidden}}
.insight{{background:#E7F0EA;border-left:4px solid var(--green);padding:16px 18px;border-radius:0 12px 12px 0;margin:16px 0;color:#233B31;line-height:1.6}}
.note{{background:#FFF8EE;border-color:#F0E4D2;padding:18px 20px;color:#765B3F;line-height:1.65;margin:16px 0}}
.note ul{{margin-bottom:0}}
.data-table{{width:100%;border-collapse:collapse;overflow:hidden;margin:16px 0;display:block;overflow-x:auto}}
.data-table th,.data-table td{{padding:10px 12px;text-align:left;border-bottom:1px solid #E8EAE4;white-space:nowrap}}
.data-table th{{font-size:.78rem;text-transform:uppercase;color:var(--muted);background:#F8F9F6}}
footer{{border-top:1px solid var(--line);margin-top:42px;padding-top:22px;color:var(--muted);font-size:.9rem}}
@media(max-width:900px){{.kpis{{grid-template-columns:repeat(2,minmax(0,1fr))}}.chart-grid{{grid-template-columns:1fr}}main{{padding:18px 14px 40px}}.hero{{padding:28px 24px}}}}
@media print{{body{{background:white}}main{{max-width:none;padding:0}}.report-section{{break-inside:avoid}}.chart-grid{{grid-template-columns:1fr 1fr}}}}
</style>
</head>
<body><main>
<header class="hero"><div class="eyebrow">Mobility intelligence · Data analysis report</div>
<h1>Shared rides.<br>Smarter journeys.</h1>
<p>An interactive analysis of commuter demand, fare behaviour, cancellations and route-level opportunities for a safer, more affordable cab-sharing experience.</p></header>
<p class="source-note">Generated from the Colab analysis datasets. Charts use the full consolidated booking data and local fare and airport data.</p>
<div class="section-kicker">AT A GLANCE · CONSOLIDATED BOOKING DATA</div>
<div class="kpis">{kpis}</div>
{"".join(sections)}
<footer><b>CABSHARE · DATA-LED PRODUCT REPORT</b><br>Charts are calculated from the included project datasets. Product ideas are hypotheses, not validated outcomes. Predictive scores are omitted until model evaluation is performed.</footer>
</main></body></html>"""


def main() -> None:
    bookings, urban, airport = load_data()
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    output = SITE_DIR / "index.html"
    output.write_text(build_html(bookings, urban, airport), encoding="utf-8")
    print(f"Generated {output}")


if __name__ == "__main__":
    main()
