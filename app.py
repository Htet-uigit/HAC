import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram

# --- PAGE CONFIG ---
st.set_page_config(page_title="HAC Student Analytics", layout="wide")

# --- SIDEBAR (No Icons) ---
with st.sidebar:
    st.header("Project Information")
    st.info("**Group:** Hierarchical Agglomerative Clustering")
    st.info("**Class:** Advanced Machine Learning Class")
    st.divider()
    st.markdown("### Deployment Guide")
    st.write("1. Fill in student profile.\n2. Enter subject scores (0-100).\n3. View hierarchical results.")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        # Loading the updated pipeline and clustered data
        pipeline = joblib.load("student_clustering_pipeline.pkl")
        train_df = pd.read_csv("student_scores_with_clusters.csv")
        
        preprocessor = pipeline.named_steps['preprocessor']
        features = ['gender', 'part_time_job', 'absence_days', 'extracurricular_activities', 
                    'weekly_self_study_hours', 'career_aspiration', 'math_score', 'history_score', 
                    'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score']
        
        X_train_proc = preprocessor.transform(train_df[features])
        
        # Calculate centroids for each cluster found in your updated notebook
        unique_clusters = sorted(train_df['cluster_label'].unique())
        centroids = {}
        for cluster in unique_clusters:
            centroids[cluster] = X_train_proc[train_df['cluster_label'] == cluster].mean(axis=0)
        
        return pipeline, train_df, centroids
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None

pipeline, train_df, centroids = load_assets()

# --- MAIN UI ---
st.title("Student Performance Clustering Dashboard")

if pipeline is not None:
    # --- STEP 1: PROFILE ---
    st.subheader("Step 1: Student Profile")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["male", "female"])
        part_time = st.selectbox("Part-time Job", [True, False])
        extra = st.selectbox("Extracurricular Activities", [True, False])
    with col2:
        absence = st.number_input("Absence Days", 0, 30, 2)
        study = st.number_input("Weekly Study Hours", 0, 168, 15)

    st.divider()
    
    # --- STEP 2: SCORES ---
    st.subheader("Step 2: Subject Scores")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        math = st.number_input("Math Score", 0, 100, 75)
        hist = st.number_input("History Score", 0, 100, 75)
    with sc2:
        phys = st.number_input("Physics Score", 0, 100, 75)
        chem = st.number_input("Chemistry Score", 0, 100, 75)
    with sc3:
        biol = st.number_input("Biology Score", 0, 100, 75)
        engl = st.number_input("English Score", 0, 100, 75)
    with sc4:
        geog = st.number_input("Geography Score", 0, 100, 75)

    # --- RUN ANALYSIS ---
    if st.button("Run Analysis and Show Result", use_container_width=True):
        try:
            # 1. Prepare Input
            input_df = pd.DataFrame([{
                'gender': gender, 'part_time_job': part_time, 'absence_days': absence,
                'extracurricular_activities': extra, 'weekly_self_study_hours': study,
                'career_aspiration': 'Unknown',
                'math_score': float(math), 'history_score': float(hist), 
                'physics_score': float(phys), 'chemistry_score': float(chem), 
                'biology_score': float(biol), 'english_score': float(engl), 
                'geography_score': float(geog)
            }])

            # 2. Process and Find Nearest Cluster
            proc = pipeline.named_steps['preprocessor']
            X_new = proc.transform(input_df)
            
            # Distance-based assignment to handle updated cluster counts
            distances = {c: np.linalg.norm(X_new - center) for c, center in centroids.items()}
            best_cluster = min(distances, key=distances.get)
            
            # Mapping based on your updated IPYNB groups
            # (Assuming Cluster 0 is High and Cluster 1 is Low based on latest draft)
            label = "High Performance Student" if best_cluster == 0 else "Low Performance Student"

            st.divider()
            
            # Result Display
            st.markdown(f"### Student Segment: {label}")
            if best_cluster == 0:
                st.success(f"Classification: {label}")
            else:
                st.warning(f"Classification: {label}")

            # --- STRICT VERTICAL VISUALIZATIONS ---
            st.subheader("Scientific Evidence")
            
            # PLOT 1: PCA (TOP)
            st.write("### 1. PCA Cluster Distribution")
            X_train_proc = proc.transform(train_df.drop(columns=['cluster_label'], errors='ignore'))
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_train_proc)
            
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=train_df['cluster_label'], cmap='viridis', alpha=0.4)
            ax1.set_xlabel("PCA Component 1")
            ax1.set_ylabel("PCA Component 2")
            st.pyplot(fig1)

            st.write("") # Spacer
            st.write("") 

            # PLOT 2: DENDROGRAM (BOTTOM)
            st.write("### 2. Hierarchical Tree Structure (Dendrogram)")
            model = pipeline.named_steps['clusterer']
            counts = np.zeros(model.children_.shape[0])
            n_samples = len(model.labels_)
            for i, merge in enumerate(model.children_):
                c = 0
                for child in merge:
                    c += 1 if child < n_samples else counts[child - n_samples]
                counts[i] = c
            linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
            
            fig2, ax2 = plt.subplots(figsize=(12, 7))
            dendrogram(linkage_matrix, truncate_mode='level', p=3, ax=ax2)
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"Analysis Error: {e}")
else:
    st.error("Assets (PKL/CSV) could not be loaded. Please ensure they are in the app directory.")
