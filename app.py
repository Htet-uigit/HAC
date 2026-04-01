import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="HAC Student Performance", layout="wide")

# --- 1. ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        pipeline = joblib.load("student_clustering_pipeline.pkl") 
        df = pd.read_csv("student_scores_with_clusters.csv")
        
        # Numeric features used in the model
        features = ['absence_days', 'weekly_self_study_hours', 'math_score', 
                    'history_score', 'physics_score', 'chemistry_score', 
                    'biology_score', 'english_score', 'geography_score']
        
        X_processed = pipeline.named_steps['preprocessor'].transform(df[features])
        X_pca_full = pipeline.named_steps['pca'].transform(X_processed)
        
        # Calculate centroids for cluster assignment since HAC lacks .predict()
        centroids_pca = {c: X_pca_full[df['cluster_label'] == c].mean(axis=0) 
                         for c in np.unique(df['cluster_label'])}
            
        return pipeline, df, centroids_pca, X_pca_full, features
    except Exception as e:
        return None, None, None, None, None

pipeline, df_clustered, centroids, X_pca, feature_cols = load_assets()

# --- 2. SIDEBAR (Project Info) ---
with st.sidebar:
    st.title("Project Details")
    st.info("**Group Name:** Hierarchical Clustering Algorithm")
    
    st.markdown("""
    ### About this Project
    This system analyzes student academic profiles using **Unsupervised Learning**. 
    It identifies patterns in study habits and test scores to group students into three distinct performance tiers.
                
    **Group members:** 
    - Thandar Htwe
    - May Mon Thant
    - May Phuu Thwel
    - Htet Myat Phone Naing
    
    ### Data Input Rules
    * **Range:** All inputs must be between **0 and 100**.
    * **Validation:** Values outside this range will be automatically rounded to the nearest limit.
    """)
    st.divider()
    st.caption("Advanced Machine Learning - Spring 2026")

# --- 3. MAIN PAGE (Title & Red Box Inputs) ---
st.title("Student Performance: 3-Cluster Analysis")
st.write("---")

if pipeline is None:
    st.stop()

# This section represents the "Red Boxes" from your reference
st.subheader("1. Data Input Features")
col1, col2, col3 = st.columns(3)

with col1:
    absence = st.number_input("Absence Days", 0, 100, 5)
    study = st.number_input("Weekly Study Hours", 0, 100, 20)
    math = st.number_input("Math Score", 0, 100, 70)

with col2:
    hist = st.number_input("History Score", 0, 100, 70)
    phys = st.number_input("Physics Score", 0, 100, 70)
    chem = st.number_input("Chemistry Score", 0, 100, 70)

with col3:
    biol = st.number_input("Biology Score", 0, 100, 70)
    engl = st.number_input("English Score", 0, 100, 70)
    geog = st.number_input("Geography Score", 0, 100, 70)

# Prediction Action
if st.button("Analyze Student & Map Cluster", use_container_width=True):
    # Prepare input
    new_row = pd.DataFrame([{
        'absence_days': absence, 'weekly_self_study_hours': study,
        'math_score': math, 'history_score': hist, 'physics_score': phys,
        'chemistry_score': chem, 'biology_score': biol, 
        'english_score': engl, 'geography_score': geog
    }])
    
    # Process
    X_new_proc = pipeline.named_steps['preprocessor'].transform(new_row)
    X_new_pca = pipeline.named_steps['pca'].transform(X_new_proc)
    
    # Distance-based assignment
    dists = {c: np.linalg.norm(X_new_pca - centroids[c]) for c in centroids}
    res_cluster = min(dists, key=dists.get)
    
    # Display Result
    st.success(f"### Student assigned to Cluster: {res_cluster}")
    
    # --- 4. VISUALIZATION ---
    st.subheader("2. Cluster Visualization (PCA Space)")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Background points
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df_clustered['cluster_label'], 
                       cmap='viridis', alpha=0.3)
    
    # The "Red X" for the current input
    ax.scatter(X_new_pca[0,0], X_new_pca[0,1], color='red', marker='*', 
               s=350, label='Picked Student')
    
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    st.pyplot(fig)
    
    # Descriptions
    desc = {
        0: "Cluster 0: Likely **High Achievers** with consistent habits.",
        1: "Cluster 1: Likely **Mid-Tier** students with balanced scores.",
        2: "Cluster 2: Likely **At-Risk** students needing academic support."
    }
    st.info(desc.get(res_cluster))
else:
    st.info("Fill in the fields above and click 'Analyze' to generate the report.")
