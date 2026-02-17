"""
Export dialogue to a Word document (.docx).
"""

import io

from docx import Document


def export_dialogue_to_docx(dialogue: list, speaker_labels: dict) -> bytes:
    """Build a Word document with the full conversation in chronological order."""
    doc = Document()
    for entry in dialogue:
        party, content = entry[0], entry[1]
        ts = entry[2] if len(entry) >= 3 else None
        label = entry[3] if len(entry) >= 4 and entry[3] else speaker_labels.get(party, party)
        p = doc.add_paragraph()
        run = p.add_run(f"{label}: ")
        run.bold = True
        p.add_run(content)
        if ts:
            doc.add_paragraph(ts.strftime("%Y-%m-%d %H:%M"))
        doc.add_paragraph()  # spacing between messages
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()
