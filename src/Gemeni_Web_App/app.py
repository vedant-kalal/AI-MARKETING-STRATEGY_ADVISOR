from agent_advisor import marketing_advice
from web_prediction import web_predict
import streamlit as st

prob, contact, previous, education, age, marital, loan = web_predict()
if st.button("Get AI Marketing Advice"):
    advice=marketing_advice(prob, contact, previous, education, age, marital, loan)
    st.markdown("# Marketing Strategy Advisor")
    st.write(advice)
