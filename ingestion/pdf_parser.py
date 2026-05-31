"""
Parses SEC filing documents (HTML, TXT, PDF) into structured text.
Critical feature: table extraction — tables are preserved as markdown,
not flattened into unreadable text. This is what separates this pipeline
from naive RAG implementations.
"""

import pdfplumber
from pathlib import Path
from loguru import logger
import re


class FilingParser:
    """
    Parses a single filing document into a list of structured sections.
    Each section has: {"section_name": str, "content": str, "content_type": "text"|"table"}
    """

    # Known SEC section headers (10-K / 10-Q)
    KNOWN_SECTIONS = [
        "Item 1.", "Item 1A.", "Item 1B.", "Item 2.", "Item 3.", "Item 4.",
        "Item 5.", "Item 6.", "Item 7.", "Item 7A.", "Item 8.", "Item 9.",
        "Item 9A.", "Item 9B.", "Item 10.", "Item 11.", "Item 12.",
        "Business", "Risk Factors", "Properties", "Legal Proceedings",
        "Management", "Financial Statements", "Quantitative",
        "Controls and Procedures", "Market for Registrant",
        "Selected Financial Data", "Liquidity", "Results of Operations",
    ]

    def parse(self, file_path: Path) -> list:
        """
        Main entry point. Routes to correct parser based on file extension.
        Returns list of section dicts.
        """
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._parse_pdf(file_path)
        elif suffix in [".htm", ".html"]:
            return self._parse_html(file_path)
        else:
            return self._parse_txt(file_path)

    def _parse_pdf(self, path: Path) -> list:
        """
        Parse PDF with pdfplumber.
        Extracts tables as markdown, text as plain paragraphs.
        """
        sections = []
        try:
            with pdfplumber.open(path) as pdf:
                current_section = "Document"
                page_texts = []

                for page_num, page in enumerate(pdf.pages):
                    # Extract tables first (before text to avoid duplication)
                    tables = page.extract_tables()
                    for table in tables:
                        if not table:
                            continue
                        md_table = self._table_to_markdown(table)
                        if md_table:
                            sections.append({
                                "section_name": current_section,
                                "content": md_table,
                                "content_type": "table",
                                "page_number": page_num + 1
                            })

                    # Extract text (excluding table bounding boxes)
                    text = page.extract_text(x_tolerance=3, y_tolerance=3)
                    if text:
                        # Detect section headers
                        detected = self._detect_section(text)
                        if detected:
                            current_section = detected

                        # Clean and add text
                        clean = self._clean_text(text)
                        if len(clean) > 50:
                            page_texts.append({
                                "section_name": current_section,
                                "content": clean,
                                "content_type": "text",
                                "page_number": page_num + 1
                            })

                sections.extend(page_texts)
        except Exception as e:
            logger.error(f"PDF parse error for {path}: {e}")
        return sections

    def _parse_html(self, path: Path) -> list:
        """
        Parse HTML SEC filings. Strip tags, detect sections, extract tables.
        Uses regex to avoid heavy dependencies.
        """
        sections = []
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")

            # Extract tables from HTML
            table_pattern = re.compile(r'<table[^>]*>(.*?)</table>', re.DOTALL | re.IGNORECASE)
            for match in table_pattern.finditer(content):
                table_html = match.group(1)
                rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL | re.IGNORECASE)
                table_data = []
                for row in rows:
                    cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, re.DOTALL | re.IGNORECASE)
                    cleaned_cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
                    if any(cleaned_cells):
                        table_data.append(cleaned_cells)
                if table_data:
                    md = self._table_to_markdown(table_data)
                    if md:
                        sections.append({
                            "section_name": "Financial Data",
                            "content": md,
                            "content_type": "table",
                            "page_number": None
                        })

            # Strip all HTML tags for text content
            clean_text = re.sub(r'<[^>]+>', ' ', content)
            clean_text = re.sub(r'&nbsp;', ' ', clean_text)
            clean_text = re.sub(r'&amp;', '&', clean_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()

            # Split into paragraphs and detect sections
            paragraphs = [p.strip() for p in clean_text.split('\n') if len(p.strip()) > 80]
            current_section = "Document"
            for para in paragraphs:
                detected = self._detect_section(para)
                if detected:
                    current_section = detected
                sections.append({
                    "section_name": current_section,
                    "content": para,
                    "content_type": "text",
                    "page_number": None
                })
        except Exception as e:
            logger.error(f"HTML parse error for {path}: {e}")
        return sections

    def _parse_txt(self, path: Path) -> list:
        """Fallback for plain text filings."""
        sections = []
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            # Remove SGML tags from EDGAR txt format
            clean = re.sub(r'<[^>]+>', ' ', content)
            clean = re.sub(r'\s+', ' ', clean).strip()
            paragraphs = [p.strip() for p in clean.split('\n') if len(p.strip()) > 80]
            current_section = "Document"
            for para in paragraphs:
                detected = self._detect_section(para)
                if detected:
                    current_section = detected
                sections.append({
                    "section_name": current_section,
                    "content": para,
                    "content_type": "text",
                    "page_number": None
                })
        except Exception as e:
            logger.error(f"TXT parse error for {path}: {e}")
        return sections

    def _table_to_markdown(self, table: list) -> str:
        """
        Convert a 2D list (rows x cols) into a markdown table string.
        Skips empty tables. First row is treated as header.
        """
        if not table or len(table) < 2:
            return ""
        # Clean cells
        cleaned = []
        for row in table:
            if row is None:
                continue
            clean_row = [str(cell).strip() if cell is not None else "" for cell in row]
            if any(c for c in clean_row):  # skip fully empty rows
                cleaned.append(clean_row)
        if len(cleaned) < 2:
            return ""

        # Normalise column count
        max_cols = max(len(r) for r in cleaned)
        padded = [r + [""] * (max_cols - len(r)) for r in cleaned]

        header = "| " + " | ".join(padded[0]) + " |"
        separator = "| " + " | ".join(["---"] * max_cols) + " |"
        rows = ["| " + " | ".join(row) + " |" for row in padded[1:]]
        return "\n".join([header, separator] + rows)

    def _detect_section(self, text: str):
        """Return section name if text starts with a known SEC section header."""
        text_upper = text.upper()
        for section in self.KNOWN_SECTIONS:
            if text_upper.startswith(section.upper()):
                return section
        return None

    def _clean_text(self, text: str) -> str:
        """Remove excessive whitespace and non-printable characters."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x20-\x7E\n]', '', text)
        return text.strip()