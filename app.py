"""Student Employment Competitiveness & Industry Skill Gap Analyzer
Beginner-friendly, clean, academic Streamlit Application.
"""
import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.data_loader import load_student_dataset, load_job_dataset, get_dataset_metadata
from src.prediction import predict_single_student, get_predictor
from src.feature_importance import get_top_contributing_factors, plot_feature_importance_figure
from src.industry_skills import get_industry_skill_analyzer
from src.improvement_priority import calculate_improvement_priorities, load_benchmark_statistics
from src.learning_recommendations import get_learning_recommendations_for_priorities
from src.report_generator import generate_pdf_report
from src.utils import get_competitiveness_badge_color

# Supported Departments (Profile Information Only)
DEPARTMENT_OPTIONS = [
    "CSE",
    "Cyber Security",
    "IT",
    "ECE",
    "EEE",
    "Mechanical",
    "Civil",
    "Other"
]

# Page Configuration
st.set_page_config(
    page_title="Student Skill Gap Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom minimal academic styling (clean light theme, no clutter)
st.markdown("""
<style>
    .main-title {
        font-size: 26px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 4px;
    }
    .sub-title {
        font-size: 14px;
        color: #475569;
        margin-bottom: 16px;
    }
    .profile-banner {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 6px;
        padding: 8px 14px;
        margin-bottom: 14px;
        font-size: 14px;
        color: #1E40AF;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .pred-badge {
        display: inline-block;
        font-size: 22px;
        font-weight: bold;
        padding: 6px 16px;
        border-radius: 6px;
        margin: 8px 0;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 6px;
        padding: 8px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "student_name" not in st.session_state:
    st.session_state.student_name = "Alex Johnson"

if "department" not in st.session_state:
    st.session_state.department = "CSE"

if "student_profile" not in st.session_state:
    st.session_state.student_profile = {
        "GPA": 3.20,
        "Credits_Completed": 90,
        "Exam_Scores": 75,
        "Attendance_Percentage": 80.0,
        "Coding_Skills": 6,
        "Communication_Skills": 6,
        "Leadership_Skills": 5,
        "Clubs_Joined": 2,
        "Events_Participated": 5,
        "Internship_Completed": 1,
        "Internship_Duration_Months": 6,
        "Library_Visits_Per_Week": 7,
        "Online_Course_Hours_Per_Week": 10
    }

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "selected_job_role" not in st.session_state:
    st.session_state.selected_job_role = ".NET Developer"

if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠 Home"

# Sidebar Navigation
st.sidebar.title("🎓 Navigation")
pages = [
    "🏠 Home",
    "👨‍🎓 Student Assessment",
    "📊 Employment Prediction",
    "💼 Industry Skill Analysis",
    "🎯 Improvement Plan",
    "📄 Report"
]

selected_page = st.sidebar.radio("Select Section:", pages, index=pages.index(st.session_state.current_page))
st.session_state.current_page = selected_page

st.sidebar.markdown("---")
st.sidebar.markdown("**Current Active Student:**")
st.sidebar.write(f"👤 **{st.session_state.student_name}**")
st.sidebar.write(f"🏛️ **Dept:** {st.session_state.department}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Source-of-Truth Datasets:**")
st.sidebar.caption("1. `College_Employment_Competitiveness.csv` (4,000 student records)")
st.sidebar.caption("2. `IT_Job_Roles_Skills.csv` (469 unique IT roles)")
st.sidebar.markdown("---")
st.sidebar.caption("Academic Pair-Programming Project | Supervised ML & Skill Gap Analysis")

analyzer = get_industry_skill_analyzer()
all_roles = analyzer.get_all_job_roles()


# ==============================================================================
# PAGE 1: HOME
# ==============================================================================
if selected_page == "🏠 Home":
    st.markdown('<div class="main-title">Student Employment Competitiveness & Industry Skill Gap Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">A transparent, data-driven academic platform for evaluating student employment competitiveness and mapping industry IT skill requirements.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### 📌 Project Overview")
        st.write("""
        This application provides an objective analysis of college student readiness and industry requirements by combining two independent empirical datasets:
        
        1. **College Employment Competitiveness Dataset**: 4,000 student records measuring academic performance, coding ability, communication, leadership, and internship tenure.
        2. **IT Job Roles and Skills Dataset**: 469 verified IT job roles specifying real-world industry responsibilities, technical skill requirements, and certifications.
        """)
        
        st.markdown("### 🎯 Core Objectives")
        st.markdown("""
        * **Answer 1: Predicted Competitiveness:** How competitive is the student's profile based on the trained Machine Learning classifier?
        * **Answer 2: Key Influencing Factors:** Which measurable features contribute most to the prediction?
        * **Answer 3: Target Role Requirements:** What specific industry skills and certifications are required for a target job role?
        * **Answer 4: Objective Improvement Plan:** Which measurable student factors should be prioritized for improvement relative to dataset benchmarks?
        """)
        
        if st.button("🚀 Start Student Assessment", use_container_width=False):
            st.session_state.current_page = "👨‍🎓 Student Assessment"
            st.rerun()

    with col2:
        st.markdown("### 📊 Dataset Overview")
        meta = get_dataset_metadata()
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Student Dataset** (`College_Employment_Competitiveness.csv`)")
        st.write(f"• **Records:** {meta['student_dataset']['rows']:,} rows")
        st.write(f"• **Features:** {len(meta['student_dataset']['columns']) - 1} numeric factors")
        st.write(f"• **Target:** `{meta['student_dataset']['target']}` ({', '.join(meta['student_dataset']['target_classes'])})")
        st.write(f"• **Missing Values:** {meta['student_dataset']['missing_values']}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Industry Job Roles Dataset** (`IT_Job_Roles_Skills.csv`)")
        st.write(f"• **Records:** {meta['job_dataset']['rows']} job postings")
        st.write(f"• **Unique IT Roles:** {meta['job_dataset']['unique_roles']} roles")
        st.write(f"• **Attributes:** Job Title, Job Description, Skills, Certifications")
        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: STUDENT ASSESSMENT
# ==============================================================================
elif selected_page == "👨‍🎓 Student Assessment":
    st.markdown('<div class="main-title">👨‍🎓 Student Assessment Form</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter student profile details and performance indicators across the 13 directly measurable dataset factors.</div>', unsafe_allow_html=True)
    
    with st.form("student_assessment_form"):
        st.subheader("1. Profile Information (Non-ML Identity Metadata)")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            student_name_input = st.text_input(
                "Student Name *",
                value=st.session_state.student_name,
                placeholder="e.g. Alex Johnson",
                help="Student's identity for the assessment report (not used in ML prediction model)."
            )
        with p_col2:
            dept_idx = DEPARTMENT_OPTIONS.index(st.session_state.department) if st.session_state.department in DEPARTMENT_OPTIONS else 0
            department_input = st.selectbox(
                "Department *",
                options=DEPARTMENT_OPTIONS,
                index=dept_idx,
                help="Academic department (profile metadata only)."
            )

        st.markdown("---")
        st.subheader("2. Academic Performance & Progression (ML Input Features)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            gpa = st.number_input("GPA (2.00 - 4.00)", min_value=2.00, max_value=4.00, value=float(st.session_state.student_profile["GPA"]), step=0.01)
        with col2:
            credits_comp = st.number_input("Credits Completed (20 - 159)", min_value=20, max_value=159, value=int(st.session_state.student_profile["Credits_Completed"]), step=1)
        with col3:
            exam_scores = st.number_input("Exam Scores (50 - 99)", min_value=50, max_value=99, value=int(st.session_state.student_profile["Exam_Scores"]), step=1)
        with col4:
            attendance = st.slider("Attendance Percentage (%)", min_value=50.0, max_value=100.0, value=float(st.session_state.student_profile["Attendance_Percentage"]), step=0.5)

        st.subheader("3. Core Skills & Extracurriculars (ML Input Features)")
        col5, col6, col7 = st.columns(3)
        with col5:
            coding_skills = st.slider("Coding Skills (Scale: 1 - 9)", min_value=1, max_value=9, value=int(st.session_state.student_profile["Coding_Skills"]))
        with col6:
            comm_skills = st.slider("Communication Skills (Scale: 1 - 9)", min_value=1, max_value=9, value=int(st.session_state.student_profile["Communication_Skills"]))
        with col7:
            lead_skills = st.slider("Leadership Skills (Scale: 1 - 9)", min_value=1, max_value=9, value=int(st.session_state.student_profile["Leadership_Skills"]))

        col8, col9 = st.columns(2)
        with col8:
            clubs_joined = st.number_input("Clubs Joined (0 - 4)", min_value=0, max_value=4, value=int(st.session_state.student_profile["Clubs_Joined"]), step=1)
        with col9:
            events_part = st.number_input("Events Participated (0 - 9)", min_value=0, max_value=9, value=int(st.session_state.student_profile["Events_Participated"]), step=1)

        st.subheader("4. Practical Experience & Self-Study (ML Input Features)")
        col10, col11, col12, col13 = st.columns(4)
        with col10:
            internship_comp = st.selectbox(
                "Internship Completed?",
                options=[1, 0],
                format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
                index=0 if st.session_state.student_profile["Internship_Completed"] == 1 else 1
            )
        with col11:
            internship_dur = st.number_input("Internship Duration (Months: 0 - 11)", min_value=0, max_value=11, value=int(st.session_state.student_profile["Internship_Duration_Months"]), step=1)
        with col12:
            library_visits = st.number_input("Library Visits / Week (0 - 14)", min_value=0, max_value=14, value=int(st.session_state.student_profile["Library_Visits_Per_Week"]), step=1)
        with col13:
            online_hours = st.number_input("Online Course Hours / Week (0 - 19)", min_value=0, max_value=19, value=int(st.session_state.student_profile["Online_Course_Hours_Per_Week"]), step=1)

        submitted = st.form_submit_button("🔍 Analyze Student Profile")

    if submitted:
        # Validation checks
        if not student_name_input or not student_name_input.strip():
            st.error("⚠️ Validation Error: Student Name cannot be empty. Please enter the student's name.")
        elif not department_input or not department_input.strip():
            st.error("⚠️ Validation Error: Department must be selected.")
        else:
            st.session_state.student_name = student_name_input.strip()
            st.session_state.department = department_input.strip()
            
            updated_profile = {
                "GPA": gpa,
                "Credits_Completed": credits_comp,
                "Exam_Scores": exam_scores,
                "Attendance_Percentage": attendance,
                "Coding_Skills": coding_skills,
                "Communication_Skills": comm_skills,
                "Leadership_Skills": lead_skills,
                "Clubs_Joined": clubs_joined,
                "Events_Participated": events_part,
                "Internship_Completed": internship_comp,
                "Internship_Duration_Months": internship_dur,
                "Library_Visits_Per_Week": library_visits,
                "Online_Course_Hours_Per_Week": online_hours
            }
            st.session_state.student_profile = updated_profile
            st.session_state.prediction_result = predict_single_student(updated_profile)
            st.success("Student profile saved and analyzed successfully! Navigating to Prediction page...")
            st.session_state.current_page = "📊 Employment Prediction"
            st.rerun()


# ==============================================================================
# PAGE 3: EMPLOYMENT PREDICTION
# ==============================================================================
elif selected_page == "📊 Employment Prediction":
    st.markdown('<div class="main-title">📊 Employment Competitiveness Prediction</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="profile-banner">👤 <strong>Student:</strong> {st.session_state.student_name} &nbsp;|&nbsp; 🏛️ <strong>Department:</strong> {st.session_state.department}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Prediction generated by the machine learning model trained on the College Employment Competitiveness dataset.</div>', unsafe_allow_html=True)
    
    # Ensure prediction is available
    if st.session_state.prediction_result is None:
        st.session_state.prediction_result = predict_single_student(st.session_state.student_profile)
        
    pred = st.session_state.prediction_result
    pred_class = pred["predicted_competitiveness"]
    badge_color = get_competitiveness_badge_color(pred_class)
    conf = pred.get("confidence_score", 0.0)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🏆 Prediction Summary")
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 13px; color: #64748B; font-weight: 600;">PREDICTED COMPETITIVENESS</div>
            <div class="pred-badge" style="background-color: {badge_color}20; color: {badge_color}; border: 1px solid {badge_color};">
                {pred_class.upper()}
            </div>
            <div style="margin-top: 10px; font-size: 14px;">
                <strong>Prediction Confidence:</strong> {conf*100:.1f}%
            </div>
            <div style="font-size: 13px; color: #475569; margin-top: 4px;">
                <strong>Model Utilized:</strong> {pred['model_name']} (selected via 5-Fold Stratified Cross-Validation)
            </div>
            <div style="margin-top: 12px; font-size: 12px; color: #64748B; font-style: italic;">
                {pred['disclaimer']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Class Probabilities Breakdown")
        probs = pred.get("probabilities", {})
        if probs:
            prob_df = pd.DataFrame({
                "Competitiveness Class": list(probs.keys()),
                "Probability": [f"{v*100:.1f}%" for v in probs.values()],
                "Score": list(probs.values())
            })
            st.dataframe(prob_df[["Competitiveness Class", "Probability"]], hide_index=True, use_container_width=True)

    with col2:
        st.markdown("### 📈 Contributing Factors")
        st.caption("Feature importances computed directly from the trained model.")
        fig = plot_feature_importance_figure(top_n=8)
        st.pyplot(fig)
        
    st.markdown("---")
    st.markdown("### 🔍 Model Explanation & Evaluation Context")
    top_5 = get_top_contributing_factors(5)
    top_str = ", ".join([f"{item[0].replace('_', ' ')} ({item[1]*100:.1f}%)" for item in top_5])
    
    st.info(f"""
    **Top Influencing Factors for this Model:** {top_str}.
    
    **Academic Notice:** Model performance is measured on this dataset and does not guarantee real-world employment outcomes. The three target classes (`High`, `Medium`, `Low`) are balanced in the empirical data.
    """)


# ==============================================================================
# PAGE 4: INDUSTRY SKILL ANALYSIS
# ==============================================================================
elif selected_page == "💼 Industry Skill Analysis":
    st.markdown('<div class="main-title">💼 Industry Skill & Job Role Analysis</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="profile-banner">👤 <strong>Student:</strong> {st.session_state.student_name} &nbsp;|&nbsp; 🏛️ <strong>Department:</strong> {st.session_state.department}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Explore IT job role expectations, required industry technical skills, and compare them against directly measurable student indicators.</div>', unsafe_allow_html=True)
    
    selected_role = st.selectbox(
        "Select Target IT Job Role (from 469 roles in IT_Job_Roles_Skills.csv):",
        options=all_roles,
        index=all_roles.index(st.session_state.selected_job_role) if st.session_state.selected_job_role in all_roles else 0
    )
    st.session_state.selected_job_role = selected_role
    
    role_info = analyzer.get_role_details(selected_role)
    alignment = analyzer.align_student_with_role(selected_role, st.session_state.student_profile)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📋 Target Role Profile")
        st.markdown(f"**Job Title:** {role_info['job_title']}")
        st.markdown(f"**Job Description:**")
        st.write(role_info['job_description'] if role_info['job_description'] else "Not available in the current dataset.")
        
        st.markdown("#### 🛠️ Required Industry Skills")
        st.caption("Extracted from IT_Job_Roles_Skills.csv (Not directly measured as individual test scores in student dataset)")
        if role_info['skills']:
            skills_html = " ".join([f"<span style='display:inline-block; background-color:#EFF6FF; color:#1D4ED8; border:1px solid #BFDBFE; border-radius:4px; padding:3px 8px; margin:3px; font-size:12px;'>{s}</span>" for s in role_info['skills']])
            st.markdown(skills_html, unsafe_allow_html=True)
        else:
            st.write("Not available in the current dataset.")
            
        st.markdown("#### 📜 Industry Certifications")
        if role_info['certifications']:
            for c in role_info['certifications']:
                st.write(f"• {c}")
        else:
            st.write("No specific certifications listed in dataset.")

    with col2:
        st.markdown("### ⚖️ Skill Alignment & Academic Distinction")
        st.markdown("""
        **Transparency Note:** The student dataset measures broad academic and behavioral indicators (e.g. *Coding Skills*, *Internship Duration*), whereas the industry dataset lists specific tools (e.g. *Python*, *Docker*). 
        """)
        
        alignment_data = alignment.get("measured_student_factors", [])
        align_df = pd.DataFrame(alignment_data)
        if not align_df.empty:
            display_align_df = align_df[["factor_name", "current_value", "concept_alignment", "measurement_status"]].rename(columns={
                "factor_name": "Student Factor",
                "current_value": "Current Value",
                "concept_alignment": "Industry Concept Alignment",
                "measurement_status": "Measurement Status"
            })
            display_align_df["Current Value"] = display_align_df["Current Value"].astype(str)
            st.dataframe(
                display_align_df,
                hide_index=True,
                use_container_width=True
            )
        
        st.caption("Industry tools (e.g. specific frameworks) are classified as *'Not directly measured by the current student dataset'*.")


# ==============================================================================
# PAGE 5: IMPROVEMENT PLAN
# ==============================================================================
elif selected_page == "🎯 Improvement Plan":
    st.markdown('<div class="main-title">🎯 Student Improvement Plan</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="profile-banner">👤 <strong>Student:</strong> {st.session_state.student_name} &nbsp;|&nbsp; 🏛️ <strong>Department:</strong> {st.session_state.department}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Objective prioritization of measurable student factors based on dataset benchmark position and model feature importance.</div>', unsafe_allow_html=True)
    
    priorities = calculate_improvement_priorities(st.session_state.student_profile)
    recommendations = get_learning_recommendations_for_priorities(priorities, top_n=4)
    
    st.markdown("### 📊 Ranked Improvement Priorities")
    st.caption("Priority Formula: Derived by combining distance below cohort median with model feature importance weights.")
    
    prio_df = pd.DataFrame(priorities)
    display_prio_df = prio_df[["rank", "feature_display", "current_value", "dataset_median", "dataset_position", "importance_level", "priority_score", "reason"]].rename(columns={
        "rank": "Priority",
        "feature_display": "Area",
        "current_value": "Current Value",
        "dataset_median": "Cohort Median",
        "dataset_position": "Dataset Position",
        "importance_level": "Model Importance",
        "priority_score": "Priority Score",
        "reason": "Analytical Reason"
    })
    
    st.dataframe(display_prio_df, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📚 Recommended Learning Areas")
    st.caption("Actionable foundational study areas mapped strictly to measurable student performance indicators.")
    
    for rec in recommendations:
        with st.expander(f"Priority {rec['rank']}: {rec['learning_area_title']} (Focus: {rec['feature_display']})", expanded=(rec['rank'] <= 2)):
            st.markdown(f"**Importance Level:** {rec['importance_level']} | **Priority Score:** {rec['priority_score']}")
            st.markdown(f"**Rationale:** {rec['reason']}")
            st.markdown("**Key Focus Areas:**")
            for area in rec["focus_areas"]:
                st.write(f"• {area}")
            st.markdown(f"**Recommended Action:** {rec['action_item']}")


# ==============================================================================
# PAGE 6: REPORT
# ==============================================================================
elif selected_page == "📄 Report":
    st.markdown('<div class="main-title">📄 Comprehensive Assessment Report</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="profile-banner">👤 <strong>Student:</strong> {st.session_state.student_name} &nbsp;|&nbsp; 🏛️ <strong>Department:</strong> {st.session_state.department}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Review complete assessment results and export downloadable PDF or JSON reports.</div>', unsafe_allow_html=True)
    
    if st.session_state.prediction_result is None:
        st.session_state.prediction_result = predict_single_student(st.session_state.student_profile)
        
    pred = st.session_state.prediction_result
    target_role = analyzer.get_role_details(st.session_state.selected_job_role)
    top_factors = get_top_contributing_factors(5)
    priorities = calculate_improvement_priorities(st.session_state.student_profile)
    recommendations = get_learning_recommendations_for_priorities(priorities, top_n=4)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Assessment Summary")
        st.write(f"• **Student Name:** **{st.session_state.student_name}**")
        st.write(f"• **Department:** **{st.session_state.department}**")
        st.write(f"• **Predicted Competitiveness:** **{pred['predicted_competitiveness'].upper()}** (Confidence: {pred['confidence_score']*100:.1f}%)")
        st.write(f"• **Target IT Role:** {target_role['job_title']}")
        st.write(f"• **Top Priority Improvement Area:** {priorities[0]['feature_display']} (Score: {priorities[0]['priority_score']})")
        
        st.markdown("#### Student Performance Indicators (13 Measured Factors)")
        prof_df = pd.DataFrame([
            {"Feature": k.replace('_', ' '), "Student Value": str(v)} for k, v in st.session_state.student_profile.items()
        ])
        st.dataframe(prof_df, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("### 📥 Export Report")
        
        # Generate PDF Bytes with Student Name & Department
        pdf_bytes = generate_pdf_report(
            student_profile=st.session_state.student_profile,
            prediction_result=pred,
            top_factors=top_factors,
            target_role=target_role,
            priorities=priorities,
            recommendations=recommendations,
            student_name=st.session_state.student_name,
            department=st.session_state.department
        )
        
        st.download_button(
            label="📄 Download Assessment PDF Report",
            data=pdf_bytes,
            file_name=f"{st.session_state.student_name.lower().replace(' ', '_')}_competitiveness_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        
        # JSON Export with Student Name & Department
        report_json_data = {
            "student_name": st.session_state.student_name,
            "department": st.session_state.department,
            "student_profile": st.session_state.student_profile,
            "prediction": pred,
            "target_role": target_role,
            "priorities": priorities,
            "recommendations": recommendations,
            "disclaimer": "Model performance is measured on this dataset and does not guarantee real-world employment outcomes."
        }
        
        st.download_button(
            label="💾 Download Assessment JSON",
            data=json.dumps(report_json_data, indent=2),
            file_name=f"{st.session_state.student_name.lower().replace(' ', '_')}_assessment_report.json",
            mime="application/json",
            use_container_width=True
        )
        
        st.markdown('<div class="metric-card" style="margin-top: 20px;">', unsafe_allow_html=True)
        st.caption("ℹ️ The PDF report includes student name, department, measured inputs, model prediction, role details, skill alignment distinctions, and improvement recommendations.")
        st.markdown('</div>', unsafe_allow_html=True)
