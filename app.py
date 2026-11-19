import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from model_utils import load_dataset, build_model, prepare_features, FEATURE_COLUMNS, evaluate_model, get_feature_importance

st.set_page_config(page_title="Titanic AI | Survival Intelligence", page_icon="🚢", layout="wide")

st.markdown("""
<style>
.main {background: #f5f7fb;}
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.hero {padding: 1.5rem 1.8rem; border-radius: 18px; color: white;
background: linear-gradient(120deg,#101c3a,#1c4b78 65%,#287b8c); margin-bottom: 1rem;}
.hero h1 {font-size: 2.2rem; margin-bottom: .3rem;}
.hero p {opacity: .9; margin: 0;}
.metric-card {background: white; padding: 1rem 1.2rem; border-radius: 14px;
border: 1px solid #e8edf4; box-shadow: 0 3px 12px rgba(20,40,80,.04);}
.small-note {color:#64748b;font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "dataset" / "train.csv"

@st.cache_data
def get_data():
    return load_dataset(DATA_PATH)

@st.cache_resource
def get_trained_model():
    df = get_data()
    return build_model(df)

try:
    data = get_data()
    model, X_test, y_test, metrics, predictions, probabilities = get_trained_model()
except Exception as e:
    st.error("The Titanic training dataset could not be loaded.")
    st.markdown("""
    **First-time setup:** Connect to the internet and click the button below to download the original Titanic training dataset.
    """)
    if st.button("Download Titanic dataset"):
        try:
            from model_utils import download_dataset
            download_dataset(DATA_PATH)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        except Exception as err:
            st.error(f"Download failed: {err}")
    st.info("If automatic download is blocked, download the original Kaggle Titanic `train.csv` and place it in the `dataset` folder with that exact filename.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.title("🚢 Titanic AI")
page = st.sidebar.radio("Navigation", [
    "Overview", "Predict Passenger", "Bulk Prediction",
    "Survival Analytics", "Model Performance", "Model Insights", "Prediction History"
])
st.sidebar.markdown("---")
st.sidebar.caption("Machine Learning: Logistic Regression")
st.sidebar.caption("Dataset: Kaggle Titanic training set")

st.markdown("""
<div class="hero">
<h1>🚢 Titanic Survival Intelligence</h1>
<p>Explore passenger data, estimate survival probability and understand a Logistic Regression model.</p>
</div>
""", unsafe_allow_html=True)

if page == "Overview":
    st.subheader("Project dashboard")
    total = len(data)
    survivors = int(data["Survived"].sum())
    deaths = total - survivors
    survival_rate = survivors / total * 100
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Passengers in dataset", f"{total:,}")
    c2.metric("Survived", f"{survivors:,}")
    c3.metric("Did not survive", f"{deaths:,}")
    c4.metric("Historical survival rate", f"{survival_rate:.1f}%")
    left,right = st.columns(2)
    with left:
        st.plotly_chart(px.pie(data, names=data["Survived"].map({0:"Did not survive",1:"Survived"}),
                               title="Survival distribution", hole=.48,
                               color_discrete_sequence=["#e07a5f","#2a9d8f"]), use_container_width=True)
    with right:
        by_class = data.groupby("Pclass", as_index=False)["Survived"].mean()
        by_class["Survival rate (%)"] = by_class["Survived"]*100
        st.plotly_chart(px.bar(by_class, x="Pclass", y="Survival rate (%)",
                               title="Survival rate by passenger class",
                               text_auto=".1f", color="Pclass"), use_container_width=True)
    st.subheader("Dataset preview")
    st.dataframe(data.head(12), use_container_width=True)
    st.caption("This is a historical dataset analysis. Model predictions are estimates, not proof of what happened to an individual passenger.")

elif page == "Predict Passenger":
    st.subheader("Individual passenger prediction")
    st.write("Enter passenger information. The model returns an estimated survival probability.")
    with st.form("passenger_form"):
        a,b,c = st.columns(3)
        pclass = a.selectbox("Passenger class", [1,2,3], index=2)
        sex = b.selectbox("Sex", ["female","male"])
        age = c.slider("Age", 0.5, 80.0, 25.0, 0.5)
        d,e,f = st.columns(3)
        sibsp = d.number_input("Siblings / spouses aboard", 0, 10, 0)
        parch = e.number_input("Parents / children aboard", 0, 10, 0)
        fare = f.number_input("Ticket fare", 0.0, 600.0, 25.0, 1.0)
        embarked = st.selectbox("Port of embarkation", ["S","C","Q"], format_func=lambda x: {"S":"Southampton","C":"Cherbourg","Q":"Queenstown"}[x])
        submitted = st.form_submit_button("Predict survival", type="primary", use_container_width=True)
    if submitted:
        row = pd.DataFrame([{"Pclass":pclass,"Sex":sex,"Age":age,"SibSp":sibsp,
                             "Parch":parch,"Fare":fare,"Embarked":embarked}])
        prob = float(model.predict_proba(prepare_features(row))[0,1])
        label = "Likely to survive" if prob >= 0.5 else "Unlikely to survive"
        st.session_state.history.append({"Passenger class":pclass,"Sex":sex,"Age":age,
            "Siblings/spouses":sibsp,"Parents/children":parch,"Fare":fare,
            "Embarked":embarked,"Survival probability (%)":round(prob*100,2),"Prediction":label})
        x,y = st.columns([1,2])
        with x:
            st.metric("Estimated survival probability", f"{prob*100:.1f}%")
            st.success(label) if prob >= .5 else st.warning(label)
        with y:
            st.plotly_chart(go.Figure(go.Indicator(
                mode="gauge+number", value=prob*100, number={"suffix":"%"},
                title={"text":"Estimated survival probability"},
                gauge={"axis":{"range":[0,100]},"bar":{"color":"#238b8e"},
                       "steps":[{"range":[0,50],"color":"#fce8e6"},{"range":[50,100],"color":"#e1f3ee"}]})),
                use_container_width=True)
        st.caption("The 50% decision threshold is a modelling choice. This probability is not a historical certainty.")

elif page == "Bulk Prediction":
    st.subheader("Bulk passenger prediction")
    st.write("Upload a CSV with these columns: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked.")
    template = pd.DataFrame([{"Pclass":3,"Sex":"male","Age":22,"SibSp":1,"Parch":0,"Fare":7.25,"Embarked":"S"}])
    st.download_button("Download CSV template", template.to_csv(index=False), "titanic_input_template.csv", "text/csv")
    upload = st.file_uploader("Upload passenger CSV", type=["csv"])
    if upload:
        try:
            batch = pd.read_csv(upload)
            missing = [c for c in FEATURE_COLUMNS if c not in batch.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                probs = model.predict_proba(prepare_features(batch))[ :, 1]
                result = batch.copy()
                result["Survival_Probability"] = np.round(probs,4)
                result["Predicted_Survived"] = (probs >= .5).astype(int)
                result["Prediction_Label"] = result["Predicted_Survived"].map({0:"Unlikely to survive",1:"Likely to survive"})
                st.success(f"Completed predictions for {len(result)} passengers.")
                st.dataframe(result, use_container_width=True)
                st.download_button("Download prediction results", result.to_csv(index=False),
                                   "titanic_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not process this CSV: {e}")

elif page == "Survival Analytics":
    st.subheader("Explore the historical Titanic dataset")
    one,two = st.columns(2)
    with one:
        st.plotly_chart(px.histogram(data, x="Age", color=data["Survived"].map({0:"Did not survive",1:"Survived"}),
                                     nbins=25, barmode="overlay", opacity=.7, title="Age distribution by outcome",
                                     labels={"color":"Outcome"}), use_container_width=True)
    with two:
        rates = data.groupby("Sex", as_index=False)["Survived"].mean()
        rates["Survival rate (%)"] = rates["Survived"]*100
        st.plotly_chart(px.bar(rates, x="Sex", y="Survival rate (%)", title="Survival rate by sex",
                               text_auto=".1f", color="Sex"), use_container_width=True)
    three,four = st.columns(2)
    with three:
        st.plotly_chart(px.box(data, x="Pclass", y="Fare", color="Survived",
                               title="Ticket fare by passenger class and outcome",
                               labels={"Survived":"Survived (0/1)"}), use_container_width=True)
    with four:
        family = data.copy()
        family["FamilySize"] = family["SibSp"] + family["Parch"] + 1
        fam = family.groupby("FamilySize", as_index=False)["Survived"].mean()
        fam["Survival rate (%)"] = fam["Survived"]*100
        st.plotly_chart(px.line(fam, x="FamilySize", y="Survival rate (%)", markers=True,
                                title="Survival rate by family size"), use_container_width=True)
    st.subheader("Filter passenger records")
    cls = st.multiselect("Passenger class", [1,2,3], default=[1,2,3])
    sex_filter = st.multiselect("Sex", ["female","male"], default=["female","male"])
    filtered = data[data["Pclass"].isin(cls) & data["Sex"].isin(sex_filter)]
    st.write(f"Showing {len(filtered)} of {len(data)} passengers")
    st.dataframe(filtered, use_container_width=True)

elif page == "Model Performance":
    st.subheader("Logistic Regression evaluation")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    m2.metric("Precision", f"{metrics['precision']:.3f}")
    m3.metric("Recall", f"{metrics['recall']:.3f}")
    m4.metric("F1-score", f"{metrics['f1']:.3f}")
    st.caption("Metrics are calculated on a held-out test split, not on the training data.")
    left,right = st.columns(2)
    with left:
        cm = metrics["confusion_matrix"]
        fig = px.imshow(cm, text_auto=True, x=["Predicted: No","Predicted: Yes"],
                        y=["Actual: No","Actual: Yes"], color_continuous_scale="Blues",
                        title="Confusion matrix")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        from sklearn.metrics import roc_curve, auc
        fpr,tpr,_ = roc_curve(y_test, probabilities)
        roc_auc = auc(fpr,tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=f"ROC AUC = {roc_auc:.3f}"))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random baseline",line={"dash":"dash"}))
        fig.update_layout(title="ROC curve",xaxis_title="False positive rate",yaxis_title="True positive rate")
        st.plotly_chart(fig,use_container_width=True)
    st.subheader("Test-set predictions")
    test_view = X_test.copy()
    test_view["Actual survival"] = y_test.values
    test_view["Predicted survival"] = predictions
    test_view["Survival probability"] = np.round(probabilities,3)
    st.dataframe(test_view.head(30), use_container_width=True)

elif page == "Model Insights":
    st.subheader("What influences the model?")
    st.write("Logistic Regression learns a coefficient for each processed feature. Positive coefficients increase the model's estimated log-odds; negative coefficients decrease them.")
    importance = get_feature_importance(model)
    st.plotly_chart(px.bar(importance, x="Coefficient", y="Feature", orientation="h",
                           color="Coefficient", color_continuous_scale="RdBu",
                           title="Logistic Regression coefficients"), use_container_width=True)
    st.info("A coefficient describes the model's learned association, not proof that a feature causes survival. One-hot encoded categories are compared with a reference category.")

else:
    st.subheader("Prediction history")
    if st.session_state.history:
        hist = pd.DataFrame(st.session_state.history)
        st.dataframe(hist, use_container_width=True)
        st.download_button("Download history CSV", hist.to_csv(index=False),
                           "titanic_prediction_history.csv", "text/csv")
        if st.button("Clear prediction history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions have been made in this session yet. Use Predict Passenger to create one.")
