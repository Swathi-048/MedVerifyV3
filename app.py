"""
MedVerify AI v3 — Flask Backend
POST /verify accepts only {image: base64} — no barcode needed
"""

import logging, datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from ai_verify import analyze

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MONGO_URI = "mongodb+srv://swathikolipaka027_db_user:H7orHQ2BdSZ4soVP@cluster0.rrnxqrs.mongodb.net/medverify?retryWrites=true&w=majority"

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    db = client.medverify
    medicines_col     = db.medicines
    verifications_col = db.verifications
    logger.info("✅ MongoDB Atlas connected")
except ConnectionFailure:
    logger.error("❌ MongoDB not reachable!")
    medicines_col = verifications_col = None

def db_ok():
    return medicines_col is not None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "db": "connected" if db_ok() else "disconnected"})

@app.route("/medicines", methods=["GET"])
def list_medicines():
    if not db_ok():
        return jsonify([])
    meds = list(medicines_col.find({}, {"_id": 0, "image": 0, "ocr_text": 0}))
    return jsonify(meds)

@app.route("/medicines", methods=["POST"])
def add_medicine():
    if not db_ok():
        return jsonify({"error": "DB unavailable"}), 503
    data = request.get_json(silent=True) or {}
    missing = [f for f in ("name", "brand", "image") if f not in data]
    if missing:
        return jsonify({"error": f"Missing: {missing}"}), 400
    barcode = data.get("barcode", data["name"].upper().replace(" ",""))
    if medicines_col.find_one({"barcode": barcode}):
        return jsonify({"error": "Already exists"}), 409
    medicines_col.insert_one({
        "barcode":  barcode,
        "name":     data["name"],
        "brand":    data["brand"],
        "dosage":   data.get("dosage", ""),
        "country":  data.get("country", "India"),
        "image":    data["image"],
        "ocr_text": data.get("ocr_text", ""),
    })
    return jsonify({"status": "inserted", "barcode": barcode}), 201

@app.route("/verify", methods=["POST"])
def verify():
    if not db_ok():
        return jsonify({"error": "DB unavailable"}), 503
    data = request.get_json(silent=True) or {}
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    all_records = list(medicines_col.find({}, {"_id": 0}))
    result = analyze(data["image"], all_records)

    # Log to DB
    try:
        verifications_col.insert_one({
            "verdict":   result.get("verdict"),
            "score":     result.get("composite_score"),
            "medicine":  result.get("identified_as"),
            "timestamp": datetime.datetime.utcnow()
        })
    except:
        pass

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
