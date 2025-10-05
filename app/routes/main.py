from flask import Blueprint, render_template, request, jsonify
import json
from app.utils import get_grade, get_radar_values
from app.util.recommendation import df_new, score_case_for_city_date, weights, case_studies
import numpy as np

from transformers import pipeline

# Load once when the app starts
text_generator = pipeline("text-generation", model="distilbert/distilgpt2")


main_blueprint = Blueprint('main', __name__)

AVAILABLE_CITIES = ["New York", "Beijing", "Delhi", "Ghaziabad", "Jaipur"]
UNAVAILABLE_CITIES = [
    "Tokyo", "London", "Paris", "Los Angeles", "Shanghai", "Mumbai", "São Paulo",
    "Cairo", "Moscow", "Seoul", "Istanbul", "Mexico City", "Bangkok", "Lagos",
    "Jakarta", "Karachi", "Sydney", "Singapore", "Berlin", "Madrid", "Toronto",
    "Rome", "Dubai", "Hong Kong", "Barcelona", "Manila", "Riyadh", "Lima",
    "Bogotá", "Chicago", "Buenos Aires", "Tehran", "Dhaka", "Johannesburg",
    "Bangalore", "Kolkata", "Kuala Lumpur", "Santiago", "Baghdad", "Hanoi",
    "Nairobi", "Lahore", "Amsterdam", "Athens", "Vienna", "Lisbon", "Stockholm",
    "Brussels", "Copenhagen", "Oslo"
]

def generate_insight(prompt):
    result = text_generator(prompt, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    # Remove the prompt from the generated text if repeated
    generated = result[0]['generated_text']
    if generated.startswith(prompt):
        generated = generated[len(prompt):].strip()
    return generated


def normalize(values):
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.5] * len(values)  # avoid division by zero if all equal
    return [(v - min_val) / (max_val - min_val) for v in values]

@main_blueprint.route('/', methods=['GET', 'POST'])
def index():
    grade = "N/A"
    selected_city = None
    selected_year = None
    selected_month = None
    recommendations = []
    radar_data = {}

    def scale_radar_values(values):
        max_vals = [150, 100, 300, 250, 20]
        return [min(v / m, 1.0) for v, m in zip(values, max_vals)]

    if request.method == 'POST':
        selected_city = request.form.get('city')
        selected_year = request.form.get('year')
        selected_month = request.form.get('month')

        if selected_city and selected_year and selected_month:
            grade = get_grade(selected_city, selected_year, selected_month)
            scaled_values = get_radar_values(selected_city, selected_year, selected_month)
            radar_data = {
                "labels": ["Avg_PM2.5_ugm3", "Avg_NO2_ppb", "Avg_AQI", "Rainfall_mm", "WindSpeed_ms"],
                "values": [float(v) for v in scaled_values]
            }
        else:
            radar_data = {
                "labels": ["Avg_PM2.5_ugm3", "Avg_NO2_ppb", "Avg_AQI", "Rainfall_mm", "WindSpeed_ms"],
                "values": [0] * 5
            }

        try:
            recs = score_case_for_city_date(selected_city, int(selected_year), int(selected_month), df_new, weights)
            recommendations = []
            for score, case_row in recs:
                recommendations.append({
                    "score": round(score, 3),
                    "case_id": case_row["CaseID"],
                    "event": case_row["Event_Name"],
                    "policy": case_row["Policy_Event"],
                    "city": case_row["City"],
                    "outcome": case_row.get("Outcome", ""),
                    "source": case_row.get("SourceReference", ""),
                    "detail_slug": case_row["CaseID"].lower()
                })
        except Exception:
            recommendations = [
                {"score": 0.35, "case_id": "CS020", "event": "New_York_Smaze", "policy": "NY Pollution Control", "detail_slug": "cs020"},
                {"score": 0.33, "case_id": "CS083", "event": "Budapest_Valley", "policy": "Heating Modernization", "detail_slug": "cs083"},
                {"score": 0.31, "case_id": "CS107", "event": "Antwerp_Port", "policy": "Emission Reduction", "detail_slug": "cs107"},
            ]

    return render_template('index.html',
                           available_cities=AVAILABLE_CITIES,
                           unavailable_cities=UNAVAILABLE_CITIES,
                           selected_city=selected_city,
                           selected_year=selected_year,
                           selected_month=selected_month,
                           grade=grade,
                           radar_data=json.dumps(radar_data),
                           recommendations=recommendations)


@main_blueprint.route('/api/recommendations')
def api_recommendations():
    city = request.args.get('city')
    year = request.args.get('year', type=int)
    month = request.args.get('month', type=int)

    if not city or not year or not month:
        return jsonify({'error': 'Missing required parameters'}), 400

    try:
        top_recs = score_case_for_city_date(city, year, month, df_new, weights)
        recs = []
        for score, row in top_recs:
            recs.append({
                'score': round(score, 3),
                'case_id': row['CaseID'],
                'event': row['Event_Name'],
                'city': row['City'],
                'policy': row['Policy_Event'],
                'outcome': row.get('Outcome', ''),
                'source': row.get('SourceReference', '')
            })
        return jsonify(recs)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main_blueprint.route('/recommendation/<slug>')
def recommendation_detail(slug):
    case = None
    for _, row in case_studies.iterrows():
        if row['CaseID'].lower() == slug.lower():
            case = row
            break
    if not case:
        return "Case study not found", 404
    return render_template('recommendation_detail.html', case=case)



from flask import request, jsonify

@main_blueprint.route('/api/generate_insight', methods=['POST'])
def generate_insight_api():
    data = request.json
    case = data.get('case')
    if not case:
        return jsonify({'error': 'No case provided'}), 400

    # Formulate prompt with case details dynamically
    prompt = (
        f"Explain the environmental impact and policy recommendations for the case:\n"
        f"Case ID: {case.get('case_id')}\n"
        f"Event: {case.get('event')}\n"
        f"Policy: {case.get('policy')}\n"
        f"Score: {case.get('score')}\n"
        f"Provide a comprehensive but concise explanation."
    )

    try:
        insight_text = generate_insight(prompt)
        return jsonify({'insight': insight_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
