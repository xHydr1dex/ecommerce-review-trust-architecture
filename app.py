import streamlit as st
import pandas as pd

# Load preprocessed dataset
df = pd.read_csv("reviews_flagged.csv")

st.title("🕵️ Suspicious Review Detector")
st.markdown("This dashboard shows reviews flagged as suspicious based on rule-based and ML-based methods.")

# Sidebar filters
st.sidebar.header("Filter options")

only_rule = st.sidebar.checkbox("Show only rule-based flagged", False)
only_ml = st.sidebar.checkbox("Show only ML-based flagged", False)
show_all = st.sidebar.checkbox("Show all suspicious reviews", True)


# Apply filters
if only_rule:
    filtered = df[(df["rule_flagged"] == True) & (df["ml_flagged"] == False)]
elif only_ml:
    filtered = df[(df["ml_flagged"] == True) & (df["rule_flagged"] == False)]
elif show_all:
    filtered = df[df["suspicious"] == True]
else:
    filtered = df.copy()  # fallback

st.subheader(f"🔍 Showing {len(filtered)} suspicious reviews")

# Display suspicious reviews
for _, row in filtered.iterrows():
    st.markdown(f"**Review ID:** {row['review_id']}")
    st.markdown(f"📝 *{row['review_text']}*")
    st.markdown(f"⭐ {row['star_rating']} | ✅ Verified: {row['verified_purchase']}")
    def colored_flag(flag):
        return f"<span style='color:green;font-weight:bold'>{flag}</span>" if flag else f"<span style='color:red;font-weight:bold'>{flag}</span>"

    st.markdown(
        f"🔖 Rule-based: {colored_flag(row['rule_flagged'])} | ML-based: {colored_flag(row['ml_flagged'])}",
        unsafe_allow_html=True
    )
    # Trust Score Progress Bar
    st.markdown("🔒 **Trust Score:**")
    trust_val = float(row["trust_score"])
    trust_val = max(0.0, min(1.0, trust_val))  # clamp between 0 and 1
    st.progress(trust_val)

    st.markdown("---")

# Show summary stats
st.sidebar.subheader("📊 Stats")
st.sidebar.markdown(f"- Total Reviews: {len(df)}")
st.sidebar.markdown(f"- Rule-flagged: {df['rule_flagged'].sum()}")
st.sidebar.markdown(f"- ML-flagged: {df['ml_flagged'].sum()}")
st.sidebar.markdown(f"- Overlap: {(df['rule_flagged'] & df['ml_flagged']).sum()}")

#.\venv\Scripts\activate   # on Windows
