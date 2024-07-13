
import streamlit as st
import altair as alt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle

# Load the TF-IDF vocabulary specific to the category
with open("toxic_vect.pkl", "rb") as f:
    tox = pickle.load(f)

with open("severe_toxic_vect.pkl", "rb") as f:
    sev = pickle.load(f)

with open("obscene_vect.pkl", "rb") as f:
    obs = pickle.load(f)

with open("insult_vect.pkl", "rb") as f:
    ins = pickle.load(f)

with open("threat_vect.pkl", "rb") as f:
    thr = pickle.load(f)

with open("identity_hate_vect.pkl", "rb") as f:
    ide = pickle.load(f)

# Load the pickled RDF models
with open("toxic_model.pkl", "rb") as f:
    tox_model = pickle.load(f)

with open("severe_toxic_model.pkl", "rb") as f:
    sev_model = pickle.load(f)

with open("obscene_model.pkl", "rb") as f:
    obs_model = pickle.load(f)

with open("insult_model.pkl", "rb") as f:
    ins_model = pickle.load(f)

with open("threat_model.pkl", "rb") as f:
    thr_model = pickle.load(f)

with open("identity_hate_model.pkl", "rb") as f:
    ide_model = pickle.load(f)

# Streamlit app
def main():
    st.title("Toxicity Prediction App")

    # Input text box for user input
    user_input = st.text_area("Enter text:")

    # Predict button
    if st.button("Predict"):
        if user_input:
            data = [user_input]

            vect = tox.transform(data)
            pred_tox = tox_model.predict_proba(vect)[:, 1]

            vect = sev.transform(data)
            pred_sev = sev_model.predict_proba(vect)[:, 1]

            vect = obs.transform(data)
            pred_obs = obs_model.predict_proba(vect)[:, 1]

            vect = thr.transform(data)
            pred_thr = thr_model.predict_proba(vect)[:, 1]

            vect = ins.transform(data)
            pred_ins = ins_model.predict_proba(vect)[:, 1]

            vect = ide.transform(data)
            pred_ide = ide_model.predict_proba(vect)[:, 1]

            out_tox = round(pred_tox[0], 2)
            out_sev = round(pred_sev[0], 2)
            out_obs = round(pred_obs[0], 2)
            out_ins = round(pred_ins[0], 2)
            out_thr = round(pred_thr[0], 2)
            out_ide = round(pred_ide[0], 2)

            st.write("Prob (Toxic):", out_tox)
            st.write("Prob (Severe Toxic):", out_sev)
            st.write("Prob (Obscene):", out_obs)
            st.write("Prob (Insult):", out_ins)
            st.write("Prob (Threat):", out_thr)
            st.write("Prob (Identity Hate):", out_ide)

            # Bar chart using altair
            categories = ['Toxic', 'Severe Toxic', 'Obscene', 'Insult', 'Threat', 'Identity Hate']
            percentages = [out_tox, out_sev, out_obs, out_ins, out_thr, out_ide]

            data_chart = pd.DataFrame({'Category': categories, 'Percentage': percentages})
            chart = alt.Chart(data_chart).mark_bar().encode(
                x='Category:N',
                y='Percentage:Q',
                tooltip=['Category', 'Percentage']
            ).properties(
                width=400
            )

            st.altair_chart(chart, use_container_width=True)

        else:
            st.warning("Please enter some text.")

 # Footer with specific padding from the bottom
    footer_html = """
    <style>
        .footer {
            
            bottom: 1cm;
            width: 100%;
            text-align: center;
        }
    </style>
    <div class="footer">
        <p><a href="https://www.linkedin.com/in/sanjeev-kumar-201bb620b/" target="_blank">Connect with me on LinkedIn @sanjeevkumar</a></p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
