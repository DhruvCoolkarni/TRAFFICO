# TrafficSense — Hybrid Traffic Pattern Discovery & Congestion Prediction

A college ML CCE prototype combining:
- **Unsupervised learning:** K-Means discovers traffic patterns without using target labels.
- **Supervised learning:** Decision Tree predicts Low / Moderate / Heavy congestion.
- **Syllabus mapping:** Unit I preprocessing, Unit II evaluation, Unit III Decision Tree, Unit IV K-Means and Elbow-method concept (3 clusters used for the MVP).

## Run on Windows (VS Code terminal)

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The app creates `data/traffic_data.csv` automatically on first run.

## Important data limitation
The included dataset is generated synthetically for demonstration. Congestion labels are generated using a transparent rule, not collected from real roads. Therefore, evaluation results demonstrate the workflow only and must not be presented as real-world accuracy.

## Input features
Vehicle count, average speed, hour, weekday/weekend, rainfall.

## Project flow
Traffic data → preprocessing → K-Means pattern discovery + Decision Tree classification → evaluation and interactive prediction.
