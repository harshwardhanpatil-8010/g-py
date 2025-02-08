from flask import Flask, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import io

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///greenscore.db'
db = SQLAlchemy(app)

class ConsumptionData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False)
    energy = db.Column(db.Float, nullable=False)
    water = db.Column(db.Float, nullable=False)
    waste = db.Column(db.Float, nullable=False)

def calculate_score(energy, water, waste):
    energy_scale = 1000
    water_scale = 500
    waste_scale = 100
    energy_score = max(0, 100 - (energy/energy_scale)*100)
    water_score = max(0, 100 - (water/water_scale)*100)
    waste_score = max(0, 100 - (waste/waste_scale)*100)
    overall = (energy_score + water_score + waste_score)/3
    return {"energy_score": energy_score, "water_score": water_score, "waste_score": waste_score, "overall_score": overall}

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    timestamp = datetime.fromisoformat(data['timestamp'])
    energy = float(data['energy'])
    water = float(data['water'])
    waste = float(data['waste'])
    record = ConsumptionData(timestamp=timestamp, energy=energy, water=water, waste=waste)
    db.session.add(record)
    db.session.commit()
    return jsonify({"status": "success"}), 201

@app.route('/score', methods=['GET'])
def score():
    record = ConsumptionData.query.order_by(ConsumptionData.timestamp.desc()).first()
    if not record:
        return jsonify({"error": "No consumption data available"}), 404
    breakdown = calculate_score(record.energy, record.water, record.waste)
    return jsonify({"timestamp": record.timestamp.isoformat(), "data": {"energy": record.energy, "water": record.water, "waste": record.waste}, "score": breakdown})

@app.route('/trend', methods=['GET'])
def trend():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    records = ConsumptionData.query.filter(ConsumptionData.timestamp >= start_date, ConsumptionData.timestamp <= end_date).order_by(ConsumptionData.timestamp).all()
    if not records:
        return jsonify({"error": "No consumption data available for trend analysis"}), 404
    data = [{"date": r.timestamp.date(), "energy": r.energy, "water": r.water, "waste": r.waste, "score": calculate_score(r.energy, r.water, r.waste)["overall_score"]} for r in records]
    df = pd.DataFrame(data)
    trend_df = df.groupby("date").mean().reset_index()
    result = trend_df.to_dict(orient='records')
    return jsonify({"trend": result})

@app.route('/recommendations', methods=['GET'])
def recommendations():
    record = ConsumptionData.query.order_by(ConsumptionData.timestamp.desc()).first()
    if not record:
        return jsonify({"error": "No consumption data available"}), 404
    breakdown = calculate_score(record.energy, record.water, record.waste)
    recs = []
    if breakdown["energy_score"] < 50:
        recs.append("Consider reducing energy usage by switching to energy-efficient appliances and lighting.")
    if breakdown["water_score"] < 50:
        recs.append("Consider reducing water consumption by fixing leaks and installing water-saving fixtures.")
    if breakdown["waste_score"] < 50:
        recs.append("Consider minimizing waste through better recycling practices and reducing single-use items.")
    if not recs:
        recs.append("Your consumption levels are excellent. Keep up the sustainable practices.")
    return jsonify({"recommendations": recs})

@app.route('/export', methods=['GET'])
def export():
    records = ConsumptionData.query.order_by(ConsumptionData.timestamp).all()
    if not records:
        return jsonify({"error": "No consumption data available"}), 404
    data = [{"timestamp": r.timestamp.isoformat(), "energy": r.energy, "water": r.water, "waste": r.waste} for r in records]
    df = pd.DataFrame(data)
    csv_data = df.to_csv(index=False)
    return send_file(io.BytesIO(csv_data.encode('utf-8')), mimetype='text/csv', as_attachment=True, download_name='consumption_data.csv')

@app.route('/predict', methods=['GET'])
def predict():
    records = ConsumptionData.query.order_by(ConsumptionData.timestamp).all()
    if len(records) < 2:
        record = ConsumptionData.query.order_by(ConsumptionData.timestamp.desc()).first()
        if not record:
            return jsonify({"error": "No consumption data available"}), 404
        breakdown = calculate_score(record.energy, record.water, record.waste)
        return jsonify({"predicted_score": breakdown["overall_score"]})
    data = [{"timestamp": r.timestamp.timestamp(), "energy": r.energy, "water": r.water, "waste": r.waste, "score": calculate_score(r.energy, r.water, r.waste)["overall_score"]} for r in records]
    df = pd.DataFrame(data)
    X = df[["timestamp", "energy", "water", "waste"]].values
    y = df["score"].values
    model = LinearRegression()
    model.fit(X, y)
    last_record = records[-1]
    future_time = last_record.timestamp.timestamp() + 86400
    X_future = np.array([[future_time, last_record.energy, last_record.water, last_record.waste]])
    prediction = model.predict(X_future)[0]
    return jsonify({"predicted_score": prediction})

if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
