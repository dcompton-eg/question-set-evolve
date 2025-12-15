"""PDF generation for question sets."""

from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

from .models import QuestionSet


def generate_question_set_pdf(question_set: QuestionSet, output_path: Path) -> None:
    """Generate a PDF document from a question set.

    Args:
        question_set: The QuestionSet to render as PDF
        output_path: Path where the PDF will be saved
    """
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=12,
        textColor=colors.HexColor('#1a1a1a'),
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=16,
        spaceAfter=8,
        textColor=colors.HexColor('#333333'),
    )

    subheading_style = ParagraphStyle(
        'CustomSubheading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor('#444444'),
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        leading=14,
    )

    question_style = ParagraphStyle(
        'QuestionText',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8,
        leading=15,
        textColor=colors.HexColor('#1a1a1a'),
    )

    metadata_style = ParagraphStyle(
        'Metadata',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#666666'),
        spaceAfter=4,
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=20,
        spaceAfter=4,
        leading=13,
    )

    story = []

    # Title
    story.append(Paragraph(question_set.title, title_style))
    story.append(Spacer(1, 0.1 * inch))

    # Metadata table
    metadata = [
        ['Target Role:', question_set.target_role],
        ['Total Time:', f'{question_set.total_time_minutes} minutes'],
        ['Questions:', str(len(question_set.questions))],
    ]

    metadata_table = Table(metadata, colWidths=[1.5 * inch, 4.5 * inch])
    metadata_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#555555')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(metadata_table)
    story.append(Spacer(1, 0.15 * inch))

    # Description
    if question_set.description:
        story.append(Paragraph('<b>Description</b>', subheading_style))
        story.append(Paragraph(question_set.description, body_style))

    story.append(Spacer(1, 0.2 * inch))

    # Questions
    story.append(Paragraph('Interview Questions', heading_style))

    for i, question in enumerate(question_set.questions, 1):
        # Question header with metadata
        difficulty_color = {
            'easy': '#28a745',
            'medium': '#ffc107',
            'hard': '#dc3545',
        }.get(question.difficulty.lower(), '#6c757d')

        header_text = (
            f'<b>Question {i}</b> '
            f'<font color="{difficulty_color}">[{question.difficulty.upper()}]</font> '
            f'<font color="#6c757d">[{question.category}] [{question.time_allocation_minutes} min]</font>'
        )
        story.append(Paragraph(header_text, subheading_style))

        # Question text
        story.append(Paragraph(question.question_text, question_style))

        # Follow-up questions
        if question.follow_up_questions:
            story.append(Paragraph('<i>Follow-up questions:</i>', metadata_style))
            for followup in question.follow_up_questions:
                story.append(Paragraph(f'• {followup}', bullet_style))

        story.append(Spacer(1, 0.15 * inch))

    # Build the PDF
    doc.build(story)
