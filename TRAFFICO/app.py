import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="TRAFFICO", page_icon="🚦", layout="wide")
DATA_PATH = "data/traffic_data.csv"
FEATURES = ["vehicle_count", "avg_speed", "hour", "is_weekend", "rainfall_mm"]
LABELS = ["Low", "Moderate", "Heavy"]
COLORS = {"Low":"#2ca25f", "Moderate":"#f0ad4e", "Heavy":"#de2d26"}

def create_dataset(n=2500, seed=42):
    rng = np.random.default_rng(seed)
    hour = rng.integers(0, 24, n)
    is_weekend = rng.integers(0, 2, n)
    rainfall = np.clip(rng.gamma(1.5, 2.0, n), 0, 20)
    rush = ((hour >= 7) & (hour <= 10)) | ((hour >= 17) & (hour <= 21))
    base = rng.normal(65, 22, n) + rush * 45 - is_weekend * 12 + rainfall * 1.3
    vehicles = np.clip(np.round(base), 5, 220).astype(int)
    speed = np.clip(rng.normal(58, 12, n) - vehicles * 0.19 - rush * 8 - rainfall * 0.45, 5, 90)
    # A transparent rule creates labels for this demonstration dataset.
    congestion_score = vehicles * 0.55 + (65 - speed) * 1.25 + rainfall * 0.7
    target = np.where(congestion_score < 75, "Low",
             np.where(congestion_score < 125, "Moderate", "Heavy"))
    return pd.DataFrame({"vehicle_count":vehicles, "avg_speed":np.round(speed,1),
                         "hour":hour, "is_weekend":is_weekend,
                         "rainfall_mm":np.round(rainfall,1), "congestion":target})

@st.cache_resource
def train_models():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        create_dataset().to_csv(DATA_PATH, index=False)
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df["congestion"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Unsupervised model: discover groups without using congestion labels.
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(X_train_scaled)
    cluster_summary = X_train.copy()
    cluster_summary["cluster"] = train_clusters
    means = cluster_summary.groupby("cluster")[["vehicle_count","avg_speed"]].mean()
    ordered = means.sort_values(["vehicle_count","avg_speed"], ascending=[True, False]).index.tolist()
    cluster_names = {ordered[0]:"Low-traffic pattern", ordered[1]:"Moderate-traffic pattern", ordered[2]:"Heavy-traffic pattern"}

    # Supervised model: learn congestion labels from labelled examples.
    tree = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced")
    tree.fit(X_train, y_train)
    predictions = tree.predict(X_test)
    return df, scaler, kmeans, tree, cluster_names, y_test, predictions, accuracy_score(y_test,predictions), confusion_matrix(y_test,predictions,labels=LABELS)

df, scaler, kmeans, tree, cluster_names, y_test, predictions, accuracy, cm = train_models()
st.title("🚦 TRAFFICO")
st.caption("Hybrid ML system: K-Means discovers traffic patterns; Decision Tree predicts congestion.")
with st.sidebar:
    st.header("Predict a traffic situation")
    vehicles = st.slider("Vehicle count", 5, 220, 120)
    speed = st.slider("Average speed (km/h)", 5, 90, 25)
    hour = st.slider("Hour of day", 0, 23, 18, format="%d:00")
    weekend = st.selectbox("Day type", ["Weekday","Weekend"])
    rain = st.slider("Rainfall (mm)", 0.0, 20.0, 2.0, 0.5)
    predict = st.button("Predict congestion", type="primary", use_container_width=True)

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Data & patterns", "🧪 Model evaluation"])
with tab1:
    st.subheader("Traffic prediction")
    sample = pd.DataFrame([[vehicles,speed,hour,int(weekend=="Weekend"),rain]],columns=FEATURES)
    pred = tree.predict(sample)[0]
    cluster_id = int(kmeans.predict(scaler.transform(sample))[0])
    cluster_name = cluster_names[cluster_id]
    c1,c2,c3 = st.columns(3)
    c1.metric("Predicted congestion", pred)
    c2.metric("Discovered cluster", f"Cluster {cluster_id}")
    c3.metric("Traffic pattern", cluster_name)
    st.info("The prediction is produced by the supervised Decision Tree. The cluster is independently assigned by unsupervised K-Means.")
    st.write("**Input summary**")
    st.dataframe(sample.rename(columns={"vehicle_count":"Vehicles","avg_speed":"Speed (km/h)","hour":"Hour","is_weekend":"Weekend (1=yes)","rainfall_mm":"Rainfall (mm)"}), use_container_width=True, hide_index=True)
    st.caption("Prototype recommendation only: this model does not use live road sensors or live maps.")
with tab2:
    st.subheader("Dataset overview")
    a,b,c = st.columns(3)
    a.metric("Total records", f"{len(df):,}")
    b.metric("Features", len(FEATURES))
    c.metric("Congestion classes", df["congestion"].nunique())
    left,right = st.columns(2)
    with left:
        fig=px.histogram(df,x="congestion",color="congestion",category_orders={"congestion":LABELS},title="Class distribution")
        st.plotly_chart(fig,use_container_width=True)
    with right:
        fig=px.scatter(df,x="vehicle_count",y="avg_speed",color="congestion",opacity=.55,title="Vehicles vs average speed")
        st.plotly_chart(fig,use_container_width=True)
    st.subheader("K-Means discovered patterns")
    cluster_plot=df[FEATURES].copy()
    cluster_plot["cluster"]=kmeans.predict(scaler.transform(df[FEATURES]))
    cluster_plot["pattern"]=cluster_plot["cluster"].map(cluster_names)
    fig=px.scatter(cluster_plot,x="vehicle_count",y="avg_speed",color="pattern",title="Traffic patterns discovered by K-Means")
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(cluster_plot.groupby(["cluster","pattern"])[["vehicle_count","avg_speed","rainfall_mm"]].mean().round(2),use_container_width=True)
with tab3:
    st.subheader("Supervised model evaluation")
    st.metric("Decision Tree test accuracy", f"{accuracy:.1%}")
    st.caption("Accuracy is measured on a held-out 20% test split. Since the demo labels are generated from a rule, this score measures how well the tree learns that synthetic rule—not real-world traffic performance.")
    cm_df=pd.DataFrame(cm,index=[f"Actual {x}" for x in LABELS],columns=[f"Predicted {x}" for x in LABELS])
    st.dataframe(cm_df,use_container_width=True)
    report=classification_report(y_test,predictions,labels=LABELS,output_dict=True,zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3),use_container_width=True)
    st.subheader("How the hybrid model works")
    st.markdown("""
    1. **Preprocessing (Unit I):** prepare numerical traffic features.
    2. **K-Means (Unit IV):** group similar traffic situations without using class labels.
    3. **Decision Tree (Unit III):** learn from labelled examples and predict Low, Moderate, or Heavy congestion.
    4. **Evaluation (Unit II):** test predictions on unseen rows using accuracy and a confusion matrix.
    """)
