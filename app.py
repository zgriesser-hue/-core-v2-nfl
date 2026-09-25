import streamlit as st

# Production dashboard router.
# Keep V2 and V3 as separate executable pages while using one Streamlit app/URL.
v2 = st.Page("app_v2.py", title="V2 Dashboard", icon="2️⃣", default=True)
v3 = st.Page("pages/1_V3_Dashboard.py", title="V3 Dashboard", icon="3️⃣")

page = st.navigation(
    {
        "NFL Model": [v2, v3],
    },
    position="sidebar",
)
page.run()
