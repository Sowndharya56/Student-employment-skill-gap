"""Comprehensive assessment report generation module (PDF, JSON, Dict)."""
import os
import io
import json
import datetime
from typing import Dict, Any, List, Optional, Tuple

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_pdf_report(
    student_name: str = "Anonymous Student",
    department: str = "Not Specified",
    student_profile: Optional[Dict[str, Any]] = None,
    prediction_result: Optional[Dict[str, Any]] = None,
    top_factors: Optional[List[Tuple[str, float]]] = None,
    target_role: Optional[Dict[str, Any]] = None,
    priorities: Optional[List[Dict[str, Any]]] = None,
    recommendations: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[str] = None,
    **kwargs
) -> bytes:
    """Generate a clean, professional PDF assessment report.
    
    Compatible with any argument ordering (positional or keyword).
    
    Args:
        student_name: Student full name (Profile metadata only).
        department: Academic department (Profile metadata only).
        student_profile: Student input dictionary (13 ML numerical factors).
        prediction_result: Output from prediction pipeline.
        top_factors: List of (feature, importance) tuples.
        target_role: Target job role details (skills, certs, description).
        priorities: Ranked improvement priorities.
        recommendations: Actionable learning recommendations.
        output_path: Optional filepath to save the PDF.
        **kwargs: Additional keyword arguments for flexible signature compatibility.
        
    Returns:
        bytes: PDF file bytes.
    """
    # 1. Flexible argument resolution for positional and keyword compatibility
    if isinstance(student_name, dict) and student_profile is None:
        # Called as generate_pdf_report(student_profile, prediction_result, ...)
        student_profile = student_name
        student_name = str(kwargs.get("student_name", "Anonymous Student"))

    # Pull any kwargs overrides if passed as keywords
    student_name = str(kwargs.get("student_name", student_name))
    department = str(kwargs.get("department", department))
    student_profile = kwargs.get("student_profile", student_profile) or {}
    prediction_result = kwargs.get("prediction_result", prediction_result) or {}
    top_factors = kwargs.get("top_factors", top_factors) or []
    target_role = kwargs.get("target_role", target_role)
    priorities = kwargs.get("priorities", priorities) or []
    recommendations = kwargs.get("recommendations", recommendations) or []
    output_path = kwargs.get("output_path", output_path)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1E293B'),
        alignment=1 # Center
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748B'),
        alignment=1
    )
    
    h2_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=10,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#334155')
    )
    
    bold_body_style = ParagraphStyle(
        'DocBodyBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#1E293B')
    )
    
    disclaimer_style = ParagraphStyle(
        'DisclaimerText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#64748B')
    )
    
    elements = []
    
    # 1. Header
    elements.append(Paragraph("Student Employment Competitiveness & Industry Skill Gap Analyzer", title_style))
    elements.append(Paragraph(f"Comprehensive Assessment Report &bull; Generated on {datetime.datetime.now().strftime('%B %d, %Y at %H:%M')}", subtitle_style))
    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563EB'), spaceBefore=2, spaceAfter=8))
    
    # 2. Student Identity & Prediction Summary Card Table
    pred_class = prediction_result.get('predicted_competitiveness', 'N/A')
    conf_score = prediction_result.get('confidence_score')
    conf_str = f"{conf_score * 100:.1f}%" if conf_score is not None else "N/A"
    model_name = prediction_result.get('model_name', 'Trained Classifier')
    
    badge_bg = colors.HexColor('#DCFCE7') if pred_class == 'High' else (colors.HexColor('#FEF3C7') if pred_class == 'Medium' else colors.HexColor('#FEE2E2'))
    badge_text_col = colors.HexColor('#166534') if pred_class == 'High' else (colors.HexColor('#92400E') if pred_class == 'Medium' else colors.HexColor('#991B1B'))
    
    pred_data = [
        [
            Paragraph("<b>Student Name:</b>", body_style),
            Paragraph(f"<b>{student_name}</b>", bold_body_style),
            Paragraph("<b>Department:</b>", body_style),
            Paragraph(f"<b>{department}</b>", bold_body_style)
        ],
        [
            Paragraph("<b>Predicted Competitiveness:</b>", body_style),
            Paragraph(f"<font color='{badge_text_col.hexval()}'><b>{pred_class.upper()}</b></font>", ParagraphStyle('PClass', parent=bold_body_style, fontSize=11, leading=13)),
            Paragraph("<b>Prediction Confidence:</b>", body_style),
            Paragraph(f"<b>{conf_str}</b>", body_style)
        ],
        [
            Paragraph("<b>Model Utilized:</b>", body_style),
            Paragraph(f"{model_name}", body_style),
            Paragraph("<b>Target Role:</b>", body_style),
            Paragraph(f"{target_role.get('job_title', 'Not Selected') if target_role else 'None Selected'}", body_style)
        ]
    ]
    
    pred_table = Table(pred_data, colWidths=[1.8*inch, 1.8*inch, 1.7*inch, 1.7*inch])
    pred_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(pred_table)
    elements.append(Spacer(1, 10))
    
    # 3. Student Profile Table (13 ML Factors)
    elements.append(Paragraph("1. Directly Measured Student Performance Indicators", h2_style))
    
    # Filter only the 13 numerical factors if profile contains other metadata
    profile_items = [(k, v) for k, v in student_profile.items() if k not in ['student_name', 'department']]
    half = (len(profile_items) + 1) // 2
    col1 = profile_items[:half]
    col2 = profile_items[half:]
    
    table_rows = [["Factor", "Value", "Factor", "Value"]]
    for i in range(half):
        f1, v1 = col1[i]
        f2, v2 = col2[i] if i < len(col2) else ("", "")
        v1_str = "Yes" if f1 == "Internship_Completed" and v1 == 1 else ("No" if f1 == "Internship_Completed" and v1 == 0 else str(v1))
        v2_str = "Yes" if f2 == "Internship_Completed" and v2 == 1 else ("No" if f2 == "Internship_Completed" and v2 == 0 else (str(v2) if f2 else ""))
        table_rows.append([
            f1.replace('_', ' '),
            v1_str,
            f2.replace('_', ' ') if f2 else "",
            v2_str
        ])
        
    prof_table = Table(table_rows, colWidths=[2.1*inch, 1.4*inch, 2.1*inch, 1.4*inch])
    prof_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(prof_table)
    elements.append(Spacer(1, 10))
    
    # 4. Target Job Role & Skills
    if target_role:
        elements.append(Paragraph("2. Target IT Job Role & Industry Skill Breakdown", h2_style))
        role_title = target_role.get("job_title", "N/A")
        role_desc = target_role.get("job_description", "N/A")
        skills_list = target_role.get("skills", [])
        certs_list = target_role.get("certifications", [])
        
        elements.append(Paragraph(f"<b>Role:</b> {role_title}", bold_body_style))
        if role_desc:
            elements.append(Paragraph(f"<b>Description:</b> {role_desc}", body_style))
            elements.append(Spacer(1, 4))
            
        skills_str = ", ".join(skills_list) if skills_list else "None listed"
        certs_str = ", ".join(certs_list) if certs_list else "None listed"
        
        elements.append(Paragraph(f"<b>Required Industry Skills (Not directly measured by student dataset):</b> {skills_str}", body_style))
        elements.append(Spacer(1, 2))
        elements.append(Paragraph(f"<b>Industry Certifications:</b> {certs_str}", body_style))
        elements.append(Spacer(1, 10))
        
    # 5. Improvement Priorities
    elements.append(Paragraph("3. Improvement Priorities (Derived from Dataset Position & Model Weights)", h2_style))
    prio_headers = ["Rank", "Measurable Area", "Current", "Dataset Position", "Importance", "Reason"]
    prio_rows = [prio_headers]
    
    for item in priorities[:6]:
        prio_rows.append([
            str(item.get("rank", "")),
            item.get("feature_display", ""),
            str(item.get("current_value", "")),
            item.get("dataset_position", ""),
            item.get("importance_level", ""),
            Paragraph(item.get("reason", ""), body_style)
        ])
        
    prio_table = Table(prio_rows, colWidths=[0.5*inch, 1.4*inch, 0.7*inch, 1.6*inch, 0.9*inch, 1.9*inch])
    prio_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563EB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(prio_table)
    elements.append(Spacer(1, 10))
    
    # 6. Learning Recommendations
    if recommendations:
        elements.append(Paragraph("4. Recommended Learning Areas", h2_style))
        for rec in recommendations[:3]:
            elements.append(Paragraph(f"<b>Priority {rec['rank']}: {rec['learning_area_title']} ({rec['feature_display']})</b>", bold_body_style))
            for f in rec.get('focus_areas', []):
                elements.append(Paragraph(f"&bull; {f}", body_style))
            elements.append(Paragraph(f"<b>Recommended Action:</b> {rec.get('action_item', '')}", body_style))
            elements.append(Spacer(1, 4))
            
    # 7. Disclaimer
    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#94A3B8'), spaceBefore=4, spaceAfter=4))
    elements.append(Paragraph("<b>Notice & Academic Disclaimer:</b> This assessment is generated based on machine learning models trained on the provided College Employment Competitiveness dataset and IT Job Roles dataset. Model predictions indicate relative readiness patterns and do not guarantee real-world employment outcomes.", disclaimer_style))
    
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
            
    return pdf_bytes
