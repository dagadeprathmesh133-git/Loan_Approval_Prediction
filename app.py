import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the trained pickle model
MODEL_PATH = "Logistic_Regression_model.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None
    print(f"Warning: {MODEL_PATH} not found. Place it in the same directory as app.py.")

# Modern Glassmorphic AI Layout HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Loan Approval Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
    
    <style>
        :root {
            --bg-dark: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --accent-cyan: #06b6d4;
            --accent-purple: #8b5cf6;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-glow: rgba(255, 255, 255, 0.1);
        }

        body {
            background: radial-gradient(circle at top left, #1e1b4b, #0f172a);
            color: var(--text-main);
            font-family: 'Plus Jakarta Sans', sans-serif;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem 0;
        }

        .glass-card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-glow);
            border-radius: 24px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
            padding: 2.5rem;
            max-width: 900px;
            width: 100%;
        }

        .header-title {
            background: linear-gradient(135deg, #38bdf8, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }

        .form-label {
            color: var(--text-muted);
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.4rem;
        }

        .form-control, .form-select {
            background-color: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #fff;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            transition: all 0.3s ease;
        }

        .form-control:focus, .form-select:focus {
            background-color: rgba(15, 23, 42, 0.8);
            border-color: var(--accent-cyan);
            box-shadow: 0 0 15px rgba(6, 182, 212, 0.3);
            color: #fff;
        }

        .btn-predict {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            border: none;
            color: #fff;
            font-weight: 600;
            padding: 0.9rem 2rem;
            border-radius: 14px;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0 10px 25px rgba(139, 92, 246, 0.3);
        }

        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 35px rgba(6, 182, 212, 0.4);
        }

        .result-box {
            display: none;
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--border-glow);
            text-align: center;
        }

        .badge-approved {
            background: rgba(34, 197, 94, 0.2);
            color: #4ade80;
            border: 1px solid #22c55e;
        }

        .badge-rejected {
            background: rgba(239, 68, 68, 0.2);
            color: #f87171;
            border: 1px solid #ef4444;
        }

        .progress {
            height: 10px;
            border-radius: 10px;
            background-color: rgba(255, 255, 255, 0.1);
        }

        .progress-bar {
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
            border-radius: 10px;
        }
    </style>
</head>
<body>

<div class="container d-flex justify-content-center">
    <div class="glass-card">
        <div class="text-center mb-4">
            <h2 class="header-title"><i class="bi bi-cpu-fill me-2"></i>AI Loan Approval Intelligence</h2>
            <p class="text-secondary small">Logistic Regression Machine Learning Assessment</p>
        </div>

        <form id="predictionForm">
            <div class="row g-3">
                <!-- Personal Info -->
                <div class="col-md-4">
                    <label class="form-label">Gender</label>
                    <select class="form-select" name="Gender" required>
                        <option value="1">Male</option>
                        <option value="0">Female</option>
                    </select>
                </div>
                <div class="col-md-4">
                    <label class="form-label">Marital Status</label>
                    <select class="form-select" name="Married" required>
                        <option value="1">Yes</option>
                        <option value="0">No</option>
                    </select>
                </div>
                <div class="col-md-4">
                    <label class="form-label">Dependents</label>
                    <select class="form-select" name="Dependents" required>
                        <option value="0">0</option>
                        <option value="1">1</option>
                        <option value="2">2</option>
                        <option value="3">3+</option>
                    </select>
                </div>

                <div class="col-md-6">
                    <label class="form-label">Education</label>
                    <select class="form-select" name="Education" required>
                        <option value="1">Graduate</option>
                        <option value="0">Not Graduate</option>
                    </select>
                </div>
                <div class="col-md-6">
                    <label class="form-label">Self Employed</label>
                    <select class="form-select" name="Self_Employed" required>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>

                <!-- Financial Info -->
                <div class="col-md-6">
                    <label class="form-label">Applicant Income ($)</label>
                    <input type="number" class="form-control" name="ApplicantIncome" placeholder="e.g. 5000" required>
                </div>
                <div class="col-md-6">
                    <label class="form-label">Coapplicant Income ($)</label>
                    <input type="number" class="form-control" name="CoapplicantIncome" placeholder="e.g. 1500" value="0" required>
                </div>

                <!-- Loan Details -->
                <div class="col-md-4">
                    <label class="form-label">Loan Amount ($ Thousands)</label>
                    <input type="number" class="form-control" name="LoanAmount" placeholder="e.g. 120" required>
                </div>
                <div class="col-md-4">
                    <label class="form-label">Term (Months)</label>
                    <select class="form-select" name="Loan_Amount_Term" required>
                        <option value="360">360 (30 Yrs)</option>
                        <option value="180">180 (15 Yrs)</option>
                        <option value="240">240 (20 Yrs)</option>
                        <option value="120">120 (10 Yrs)</option>
                        <option value="84">84 (7 Yrs)</option>
                    </select>
                </div>
                <div class="col-md-4">
                    <label class="form-label">Credit History</label>
                    <select class="form-select" name="Credit_History" required>
                        <option value="1">Good (1.0)</option>
                        <option value="0">Bad (0.0)</option>
                    </select>
                </div>

                <div class="col-md-12">
                    <label class="form-label">Property Area</label>
                    <select class="form-select" name="Property_Area" required>
                        <option value="2">Urban</option>
                        <option value="1">Semiurban</option>
                        <option value="0">Rural</option>
                    </select>
                </div>

                <div class="col-12 mt-4">
                    <button type="submit" class="btn btn-predict">
                        <i class="bi bi-magic me-2"></i>Analyze Eligibility
                    </button>
                </div>
            </div>
        </form>

        <div id="resultBox" class="result-box">
            <h4 class="mb-2">Assessment Result</h4>
            <span id="resultBadge" class="badge px-3 py-2 fs-6 rounded-pill mb-3"></span>
            
            <div class="mt-3">
                <div class="d-flex justify-content-between mb-1 small text-muted">
                    <span>Approval Confidence</span>
                    <span id="probText">0%</span>
                </div>
                <div class="progress">
                    <div id="probBar" class="progress-bar" role="progressbar" style="width: 0%"></div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
    document.getElementById('predictionForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        const resultBox = document.getElementById('resultBox');
        const resultBadge = document.getElementById('resultBadge');
        const probText = document.getElementById('probText');
        const probBar = document.getElementById('probBar');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.status === 'success') {
                resultBox.style.display = 'block';
                resultBadge.innerText = data.result === 'Approved' ? 'Loan Approved' : 'Loan Rejected';
                resultBadge.className = `badge px-3 py-2 fs-6 rounded-pill mb-3 ${data.result === 'Approved' ? 'badge-approved' : 'badge-rejected'}`;
                
                probText.innerText = `${data.probability}%`;
                probBar.style.width = `${data.probability}%`;
            } else {
                alert('Error: ' + data.message);
            }
        } catch (err) {
            alert('Failed to connect to backend server.');
        }
    });
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"status": "error", "message": "Model file missing on server."}), 500

    try:
        # Extract features matching model order: 
        # ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
        #  'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
        #  'Credit_History', 'Property_Area']
        gender = float(request.form.get("Gender", 0))
        married = float(request.form.get("Married", 0))
        dependents = float(request.form.get("Dependents", 0))
        education = float(request.form.get("Education", 0))
        self_employed = float(request.form.get("Self_Employed", 0))
        applicant_income = float(request.form.get("ApplicantIncome", 0))
        coapplicant_income = float(request.form.get("CoapplicantIncome", 0))
        loan_amount = float(request.form.get("LoanAmount", 0))
        loan_amount_term = float(request.form.get("Loan_Amount_Term", 360))
        credit_history = float(request.form.get("Credit_History", 1))
        property_area = float(request.form.get("Property_Area", 0))

        features = np.array([[
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount,
            loan_amount_term, credit_history, property_area
        ]])

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
        approval_prob = round(float(probabilities[1]) * 100, 2)

        result_text = "Approved" if prediction == 1 else "Rejected"

        return jsonify({
            "status": "success",
            "prediction": int(prediction),
            "result": result_text,
            "probability": approval_prob
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
