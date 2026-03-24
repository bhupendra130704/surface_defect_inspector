import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title = 'Metal Defect Inspector',
    page_icon = '🏭',
    layout = 'wide'
)
st.title('🏭Metal Surface Defect Inspection Dashboard')
st.caption('NEU Surface Defect Dataset . YOLOv8m . 74.7% mAP@50')

@st.cache_data(ttl = 5)
def load_data():
    conn = sqlite3.connect('Requirement_files/defect_log.db')
    df = pd.read_sql_query("SELECT * FROM inspections", conn)
    conn.close()
    return df

df = load_data()

if len(df) == 0:
    st.error("No data found")
    st.stop

st.subheader("SUMMARY")
col1,col2,col3,col4 = st.columns(4)

col1.metric("Total Detections", len(df))
col2.metric("Auto-Rejected", len(df[df['decision'] == 'REJECT']))
col3.metric("Flagged", len(df[df['decision'] == 'FLAG']))
col4.metric("Avg Confidence", f"{df['confidence'].mean():.2f}")

st.divider()

#### Charts
left,right = st.columns(2)
with left:
    st.subheader("Defect Type Breakdown")
    defect_counts = df['defect'].value_counts()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(defect_counts.index, defect_counts.values,
           color=['#00e5ff','#ff5e3a','#34d399','#fbbf24','#a78bfa','#f472b6'])
    ax.set_xlabel('Defect Type')
    ax.set_ylabel('Count')
    ax.tick_params(axis = 'x', rotation =30)
    plt.tight_layout()
    st.pyplot(fig)

with right:
    st.subheader("Decision Breakdown")
    decision_counts = df['decision'].value_counts()
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    colors = ['#ff5e3a' if d == 'REJECT' else '#fbbf24'
              for d in decision_counts.index]
    ax2.pie(decision_counts.values,
            labels=decision_counts.index,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=False)

st.divider()

## confidence chat
st.subheader("Avg COnfidence per Defect Type")
conf_by_defect = df.groupby('defect')['confidence'].mean().sort_values(ascending=False)
fig3,ax3 = plt.subplots(figsize=(10,3))
ax3.barh(conf_by_defect.index, conf_by_defect.values, color='#00e5ff')
ax3.set_xlabel('Average Confidence')
ax3.axvline(x=0.5, color='red', linestyle='--', label='Threshold 0.5')
ax3.axvline(x=0.7, color='orange', linestyle='--', label='Threshold 0.7')
ax3.legend()
plt.tight_layout()
st.pyplot(fig3)

st.subheader("Inspection Log")
st.dataframe(
    df.sort_values('id', ascending=False).head(20),
    use_container_width = True
)

##reject log
st.subheader("Rejected Parts Only")
rejects = df[df['decision'] == 'REJECT'].sort_values('id', ascending=False)
st.dataframe(rejects,use_container_width=True)