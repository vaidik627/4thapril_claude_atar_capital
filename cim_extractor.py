"""
CIM (Confidential Information Memorandum) Document Parser & Extraction System
Targeted for large (80-150+ pages) financial PDFs. Extracts text and tables,
filters for financial sections, and utilizes an LLM to reliably extract 
fields based on variable client requirements.
"""

import os
import re
import json
import logging
import pdfplumber
import pandas as pd
from typing import List, Dict, Any, Optional
import argparse

# Load .env file if present
def _load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

_load_env()

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CIMParser:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.extracted_text = []
        self.extracted_tables = []
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

    def parse_pdf(self) -> None:
        """
        Parses the entire PDF, extracting both text blocks and tabular structures page by page.
        Suitable for 80-150 page documents.
        """
        logging.info(f"Opening PDF: {self.pdf_path} for parsing...")
        
        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logging.info(f"Total pages detected: {total_pages}. Starting extraction...")
            
            for index, page in enumerate(pdf.pages):
                page_num = index + 1
                
                # 1. Text Extraction
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        self.extracted_text.append({
                            "page": page_num,
                            "text": text.strip()
                        })
                except Exception as e:
                    logging.warning(f"Failed to extract text on page {page_num}: {e}")

                # 2. Table Extraction
                try:
                    tables = page.extract_tables()
                    for t_idx, table in enumerate(tables):
                        if table and len(table) > 1:
                            # Use Pandas to structure it
                            headers = table[0]
                            # Clean up empty or None headers to prevent Pandas errors
                            headers = [str(col).strip() if col else f"Col_{i}" for i, col in enumerate(headers)]
                            
                            df = pd.DataFrame(table[1:], columns=headers)
                            df = df.fillna("")  # Remove None values
                            
                            self.extracted_tables.append({
                                "page": page_num,
                                "table_index": t_idx + 1,
                                "data": df.to_dict(orient="records")
                            })
                except Exception as e:
                    logging.warning(f"Failed to extract tables on page {page_num}: {e}")

                if page_num % 20 == 0 or page_num == total_pages:
                    logging.info(f"Processed {page_num}/{total_pages} pages...")

        logging.info("PDF parsing completed successfully.")

    def find_financial_sections(self, keywords: List[str] = None) -> List[Dict[str, Any]]:
        """
        Filters the extracted PDF content, returning only pages/sections that contain
        important financial keywords. This reduces LLM token usage and prevents hallucinations.
        """
        if not keywords:
            keywords = [
                "revenue", "total revenue", "net revenue",
                "gross margin", "total gross margin",
                "contribution margin", "gross profit",
                "sg&a", "sga", "selling general", "selling, general",
                "general & administrative", "general and administrative",
                "operating expenses", "total operating expenses",
                "overhead", "opex",
                "operating income", "operating profit", "income from operations", "ebit",
                "depreciation", "amortization", "d&a",
                "adjusted ebitda", "adj. ebitda", "adj ebitda",
                "reported ebitda", "normalized ebitda",
                "ebitda adjustments", "management adjustments",
                "income statement", "financial performance",
                "historical financials", "projected financials",
                "other expense", "other income", "other (income)", "other income/expense",
                "other income / expenses", "non-operating expense", "non-operating income",
                "interest expense", "interest expense, net", "net interest expense",
                "interest and debt expense", "finance costs", "finance charges",
                "interest charges", "interest on debt",
                "capex", "capital expenditure", "capital expenditures", "total capex",
                "capital spending", "purchases of property", "pp&e",
                "change in working capital", "change in nwc", "changes in working capital",
                "working capital change", "(growth) / decline in net working capital",
                "increase in working capital", "decrease in working capital",
                "change in operating working capital",
            ]
            
        logging.info(f"Filtering content over {len(self.extracted_text)} text pages and {len(self.extracted_tables)} tables...")
        relevant_content = []
        relevant_pages = set()
        
        # Keyword matching on text
        for text_block in self.extracted_text:
            text_lower = text_block["text"].lower()
            if any(kw in text_lower for kw in keywords):
                relevant_content.append({
                    "type": "text", 
                    "page": text_block["page"], 
                    "content": text_block["text"].replace('\n', ' ')
                })
                relevant_pages.add(text_block["page"])
                
        # Retrieve all tables present on relevant pages OR containing financial keywords in their data
        for table_block in self.extracted_tables:
            table_str = json.dumps(table_block["data"]).lower()
            if table_block["page"] in relevant_pages or any(kw in table_str for kw in keywords):
                relevant_content.append({
                    "type": "table", 
                    "page": table_block["page"], 
                    "content": table_block["data"]
                })

        # Sort by page number to maintain reading order
        relevant_content.sort(key=lambda x: x["page"])
        logging.info(f"Identified {len(relevant_content)} relevant standalone sections (text/tables).")
        return relevant_content

class LLMExtractor:
    def __init__(self, api_key: str, provider: str = "openai", model_name: str = "gpt-4o"):
        self.api_key = api_key
        self.provider = provider.lower()
        self.model_name = model_name
        
        if self.provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
        elif self.provider == "nvidia":
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=self.api_key
                )
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
        elif self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install anthropic: pip install anthropic")
        else:
            raise ValueError("Unsupported provider. Choose 'openai', 'anthropic', or 'nvidia'.")

    def map_to_excel_columns(self, extracted_data: Dict) -> Dict:
        """
        Converts extracted year labels (e.g. '2021_A', '2024_E', 'TTM_2024') into
        Excel template column labels (FY19, FY20, FY21, Year_1 ... Year_5).

        Logic:
          - Historical (_A) years: sort ascending, map the last 3 to FY19 → FY20 → FY21
            (FY21 = most recent historical year, FY19 = oldest of the 3)
          - TTM: treated as an extra historical data point (slotted after FY21 if present,
            otherwise appended as a note key 'TTM')
          - Projected (_E) years: sort ascending, map up to 5 years to Year_1 … Year_5
        """
        HIST_LABELS = ["FY19", "FY20", "FY21"]
        PROJ_LABELS = ["Year_1", "Year_2", "Year_3", "Year_4", "Year_5"]

        excel_mapped = {}

        for field, periods in extracted_data.items():
            if not isinstance(periods, dict):
                excel_mapped[field] = periods
                continue

            historical: Dict[int, Any] = {}
            projected:  Dict[int, Any] = {}
            ttm_value = None

            for period_key, value in periods.items():
                key_upper = period_key.upper()
                year_match = re.search(r'(\d{4})', period_key)
                year = int(year_match.group(1)) if year_match else None

                if "TTM" in key_upper:
                    ttm_value = value
                elif key_upper.endswith("_A") and year:
                    historical[year] = value
                elif key_upper.endswith("_E") and year:
                    projected[year] = value
                elif year:
                    # Fallback: guess by year — ≤ current year = historical, else projected
                    if year <= 2024:
                        historical[year] = value
                    else:
                        projected[year] = value

            mapped: Dict[str, Any] = {}

            # Map historical → FY19 / FY20 / FY21
            sorted_hist = sorted(historical.items())          # [(year, val), ...]
            recent_hist = sorted_hist[-3:]                    # keep latest 3
            for i, label in enumerate(HIST_LABELS):
                offset = i - (3 - len(recent_hist))          # align to rightmost slot
                if offset >= 0:
                    mapped[label] = recent_hist[offset][1]
                else:
                    mapped[label] = None

            # TTM gets its own key (informational, not mapped to a column)
            if ttm_value is not None:
                mapped["TTM"] = ttm_value

            # Map projected → Year_1 … Year_5
            sorted_proj = sorted(projected.items())
            for i, label in enumerate(PROJ_LABELS):
                if i < len(sorted_proj):
                    mapped[label] = sorted_proj[i][1]
                else:
                    mapped[label] = None

            excel_mapped[field] = mapped

        return excel_mapped

    def extract_fields(self, relevant_content: List[Dict[str, Any]], client_requirements: str) -> Optional[Dict]:
        """
        Sends the filtered context and client instruction to the LLM to dynamically extract JSON.
        """
        # Format context for the prompt securely
        context_str = ""
        for item in relevant_content:
            context_str += f"\n--- Page {item['page']} ({item['type'].upper()}) ---\n"
            if item["type"] == "table":
                context_str += json.dumps(item["content"], indent=2) + "\n"
            else:
                context_str += str(item["content"]) + "\n"
                
        # To avoid exceeding max bounds, truncate if ridiculously large. (E.g. limit to 100k chars ~25k tokens roughly)
        if len(context_str) > 300000:
            logging.warning("Context is overly large, truncating cautiously...")
            context_str = context_str[:300000] + "\n...[TRUNCATED]"

        system_prompt = (
            "You are an elite financial M&A analyst. Your task is to process excerpts "
            "from a Confidential Information Memorandum (CIM) to extract financial metrics.\n"
            "You will be given unstructured text and semi-structured tabular data.\n\n"
            "CRITICAL RULES:\n"
            "1. Output exactly ONE valid JSON object.\n"
            "2. Ensure Keys inside JSON perfectly map what the client requested.\n"
            "3. Do NOT hallucinate data. If a requested metric/year/value is NOT present in the context, explicitly use null (no quotes).\n"
            "4. Distinguish carefully between historical (actual) vs projected (estimated/forecast) metrics.\n"
            "5. OUTPUT UNITS: Always output all numeric values in THOUSANDS ($000s). "
            "   If the document reports in millions, multiply by 1000. "
            "   If in actual dollars, divide by 1000. "
            "   Check every table header (e.g. '$ in thousands', '$ in millions') before outputting.\n"
            "6. YEAR LABEL FORMAT: Use exactly this convention for period keys:\n"
            "   - Historical/Actual years: 'YYYY_A' (e.g. '2021_A', '2022_A', '2023_A')\n"
            "     A suffix in the document (e.g. '25A', 'Dec-25A', '2025A') always means ACTUAL — label it YYYY_A regardless of how recent the year is.\n"
            "   - Projected/Estimated/Forecast years: 'YYYY_E' (e.g. '2024_E', '2025_E')\n"
            "     Only use _E if the document explicitly marks the period as E, F, P, Est., Proj., Forecast, or Budget.\n"
            "   - Budget years: also use 'YYYY_E' (e.g. '2025_B' in document → output as '2025_E')\n"
            "   - Trailing Twelve Months: 'TTM_YYYY' (e.g. 'TTM_2024')\n"
            "   - Do NOT use '2023A', '2024E', 'FY2023' or any other format.\n"
            "7. EXTRACT FROM TABLES ONLY: Only extract numeric values from structured financial tables (P&L tables, income statement tables). "
            "   Do NOT extract numbers from charts, graphs, bullet points, narrative text, or illustrative figures. "
            "   If a year appears only in a chart or narrative but not in a financial table, do NOT include it.\n"
            "8. Check the units of each table independently. If one table uses thousands and another uses millions, normalise both to thousands before outputting.\n"
            "9. SG&A FIELD ALIASES: The SG&A field may appear under many different names depending on the document. "
            "   Treat ALL of the following as SG&A and map them to 'SG&A' in your output:\n"
            "   - 'SG&A', 'SGA', 'Selling, General & Administrative', 'Selling, General and Administrative'\n"
            "   - 'G&A', 'General & Administrative', 'General and Administrative'\n"
            "   - 'Operating Expenses', 'Total Operating Expenses', 'Overhead', 'Opex'\n"
            "   - Any expense line that sits BELOW gross margin / contribution margin and ABOVE EBITDA.\n"
            "   IMPORTANT: Do NOT confuse SG&A with COGS or direct costs — those are already captured in Gross Margin.\n"
            "10. OPERATING INCOME EXTRACTION LOGIC:\n"
            "   OUTPUT FORMAT per year: {\"value\": number_or_null, \"source\": \"extracted\"|\"calculated\"|\"null\", \"formula\": null_or_string, \"confidence\": \"high\"|\"medium\"|\"low\"}\n"
            "   If data is insufficient for a year → output null (not an object) for that year.\n\n"
            "   STRICT PRIORITY — attempt each step in order, stop at the first that succeeds:\n\n"
            "   STEP 1 — DIRECT EXTRACTION (confidence=high):\n"
            "     Scan every financial P&L table for a row explicitly labeled EXACTLY one of:\n"
            "     'Operating Income', 'Operating Profit', 'EBIT', 'Income from Operations'\n"
            "     Rules: The label must be an exact row header in a structured table — NOT a chart title, NOT a footnote, NOT narrative text.\n"
            "     If found → extract the value, source='extracted', formula=null, confidence='high'.\n\n"
            "   STEP 2 — CALCULATE: EBITDA minus D&A (confidence=medium):\n"
            "     Only attempt if STEP 1 failed.\n"
            "     Conditions — ALL must be true in the SAME table:\n"
            "       a) A row labeled 'EBITDA' (NOT 'Adjusted EBITDA', NOT 'Adj. EBITDA', NOT 'Pro Forma EBITDA', NOT 'PF EBITDA') is present.\n"
            "          If the table has BOTH 'EBITDA' and 'Adjusted EBITDA', use only the plain 'EBITDA' row.\n"
            "          If ONLY 'Adjusted EBITDA' or 'Adj. EBITDA' exists → SKIP this step entirely.\n"
            "       b) A row labeled 'Depreciation', 'D&A', 'Depreciation & Amortization', or 'Depreciation and Amortization' is present.\n"
            "     If both conditions met → Operating Income = EBITDA - D&A.\n"
            "     source='calculated', formula='EBITDA - D&A', confidence='medium'.\n\n"
            "   STEP 3 — CALCULATE: Gross Profit minus Operating Expenses (confidence=medium):\n"
            "     Only attempt if STEP 1 and STEP 2 both failed.\n"
            "     Conditions — ALL must be true in the SAME table:\n"
            "       a) A row labeled 'Gross Profit', 'Gross Margin', or 'Contribution Margin' is present.\n"
            "       b) A row labeled 'Operating Expenses', 'Total Operating Expenses', 'SG&A', or 'G&A' is present.\n"
            "       c) These are in the SAME table — never combine rows from different tables.\n"
            "     If met → Operating Income = Gross Profit - Operating Expenses.\n"
            "     source='calculated', formula='Gross Profit - OpEx', confidence='medium'.\n\n"
            "   STEP 4 — CALCULATE: Revenue minus COGS minus Operating Expenses (confidence=low):\n"
            "     Only attempt if STEP 1, 2, and 3 all failed.\n"
            "     Conditions — ALL must be in the SAME table:\n"
            "       Revenue, COGS (or Cost of Revenue / Direct Costs), and Operating Expenses.\n"
            "     If met → Operating Income = Revenue - COGS - Operating Expenses.\n"
            "     source='calculated', formula='Revenue - COGS - OpEx', confidence='low'.\n\n"
            "   HARD RULES — violation means return null:\n"
            "     - NEVER use Adjusted EBITDA, Pro Forma EBITDA, or any EBITDA variant with a qualifier.\n"
            "     - NEVER mix values from different tables or different pages.\n"
            "     - NEVER assume or estimate a missing component — if any required input is missing, skip that step.\n"
            "     - NEVER extract from charts, graphs, dashboards, or bullet points.\n"
            "     - Convert parentheses to negative: (1,200) = -1200.\n"
            "     - If a year's value results in an implausible number (e.g. Operating Income > Revenue), return null.\n"
            "     - If none of the 4 steps succeed for a year → output null for that year (not an empty object).\n"
            "11. ADJ. EBITDA EXTRACTION LOGIC:\n"
            "   OUTPUT: flat numeric value per year (same format as Total_Revenue). Use null if not found.\n\n"
            "   PRIORITY ORDER — attempt each step in order, stop at first success:\n\n"
            "   STEP 1 — DIRECT EXTRACTION (preferred):\n"
            "     Scan P&L tables for a row explicitly labeled one of (in this priority):\n"
            "       a) 'Adjusted EBITDA' or 'Adj. EBITDA' or 'Adj EBITDA'\n"
            "       b) 'Management Adjusted EBITDA' or 'Mgmt. Adj. EBITDA'\n"
            "       c) 'Normalized EBITDA'\n"
            "       d) 'EBITDA' (plain, no qualifier) — ONLY if no adjusted variant exists anywhere in the document\n"
            "     Rules:\n"
            "       - If BOTH 'Adjusted EBITDA' AND 'PF Adjusted EBITDA' (Pro Forma) exist → use 'Adjusted EBITDA' (NOT Pro Forma).\n"
            "       - If ONLY 'PF Adjusted EBITDA' or 'Pro Forma Adjusted EBITDA' exists → use it (it is the best available adjusted figure).\n"
            "       - If a table is labeled 'Pro Forma Adjusted Income Statement' — this contains the Adj. EBITDA figures, extract from there.\n"
            "       - NEVER extract from charts, dashboards, or bullet points — only structured financial tables.\n\n"
            "   STEP 2 — CALCULATE: Reported EBITDA + Total Management Adjustments:\n"
            "     Only attempt if Step 1 failed.\n"
            "     Conditions — ALL must be true in the SAME table:\n"
            "       a) A row labeled 'Reported EBITDA', 'EBITDA' (unadjusted), or 'EBITDA before adjustments' is present.\n"
            "       b) A row labeled 'Total Management Adjustments', 'Total Adjustments', or 'Total Add-backs' is present.\n"
            "     → Adj. EBITDA = Reported EBITDA + Total Management Adjustments.\n"
            "     Note: Adjustments are typically positive (add-backs); verify sign before calculating.\n\n"
            "   HARD RULES:\n"
            "     - NEVER use Operating Income or Net Income as a proxy for Adj. EBITDA.\n"
            "     - NEVER mix values from different tables or pages.\n"
            "     - Convert parentheses to negative: (1,200) = -1200.\n"
            "     - Units: normalise to $000s. Check table header for '$ in thousands' vs '$ in millions' vs 'CAD millions'.\n"
            "       If currency is CAD (Canadian dollars), still output the number as-is in thousands — do NOT convert to USD.\n"
            "     - If a document has multiple EBITDA adjustment schedules (e.g. management + synergy), use ONLY the\n"
            "       management-adjusted figure, NOT the synergy/pro forma layer unless it is the only one available.\n"
            "     - If a year has no extractable or calculable Adj. EBITDA → output null.\n"
            "12. EBITDA (REPORTED/PLAIN) EXTRACTION LOGIC:\n"
            "   This is the unadjusted/reported EBITDA — BEFORE any management add-backs or adjustments.\n"
            "   OUTPUT: flat numeric value per year. Use null if not found.\n\n"
            "   EXTRACTION RULES:\n"
            "     STEP 1 — Look for a row explicitly labeled one of (in priority):\n"
            "       a) 'Reported EBITDA'\n"
            "       b) 'EBITDA' (plain, with no qualifier like Adjusted/Normalized/Pro Forma)\n"
            "       c) 'EBITDA before adjustments' or 'EBITDA before add-backs'\n"
            "     STEP 2 — If no plain EBITDA row exists but an adjustment table shows 'EBITDA' as the starting\n"
            "       line before listing adjustments → extract that value as Reported EBITDA.\n"
            "   HARD RULES:\n"
            "     - NEVER use Adjusted EBITDA, Adj. EBITDA, Normalized EBITDA, or PF EBITDA for this field.\n"
            "     - NEVER use Operating Income or Net Income as a proxy.\n"
            "     - NEVER extract from charts, dashboards, or bullet points — only structured financial tables.\n"
            "     - NEVER mix values from different tables.\n"
            "     - Convert parentheses to negative: (1,200) = -1200.\n"
            "     - Normalise units to $000s. If CAD, output as-is without USD conversion.\n"
            "     - If a document only has Adjusted EBITDA and no plain EBITDA → output null for this field.\n"
            "     - If a year has no plain/reported EBITDA → output null for that year.\n"
            "13. OTHER EXPENSE / (INCOME) EXTRACTION LOGIC:\n"
            "   This is the non-operating 'Other expense / (income)' line (Excel row 21).\n"
            "   OUTPUT: flat numeric value per year. Use null if not found.\n\n"
            "   EXTRACTION RULES:\n"
            "     STEP 1 — Look for a row explicitly labeled one of (case-insensitive):\n"
            "       a) 'Other expense / (income)' or 'Other (income) / expense'\n"
            "       b) 'Other income / expenses' or 'Other income/expense'\n"
            "       c) 'Other income, net' or 'Other expense, net'\n"
            "       d) 'Non-operating expense' or 'Non-operating income'\n"
            "       e) 'Other income (expense)' or 'Other expense (income)'\n"
            "     STEP 2 — Also check EBITDA adjustment schedules: if a line explicitly labeled\n"
            "       'Other Income / Expenses' or 'Other expense' appears inside the adjustment table, extract it.\n"
            "   HARD RULES:\n"
            "     - Extract ONLY if the row is explicitly labeled as above in a structured financial table.\n"
            "     - NEVER extract if the label also includes 'interest' (e.g. 'Interest and other expense') —\n"
            "       that is Interest Expense, not Other Expense.\n"
            "     - NEVER derive or calculate this field from other lines.\n"
            "     - NEVER extract from charts, dashboards, bullet points, or narrative text.\n"
            "     - Sign convention: expense shown in parentheses = negative number.\n"
            "       If shown as positive (no parentheses) in an expense section → keep positive.\n"
            "     - Convert parentheses to negative: (1,296) = -1296.\n"
            "     - Normalise units to $000s. If CAD, output as-is without USD conversion.\n"
            "     - Extract ONLY for years where the value is explicitly stated in the table.\n"
            "     - If the line is not present in the document → output null for all years.\n"
            "     - If a year has no value in an otherwise present row → output null for that year.\n"
            "14. INTEREST EXPENSE EXTRACTION LOGIC:\n"
            "   This is the interest paid on debt — a non-operating P&L line (Excel row 22).\n"
            "   OUTPUT: flat numeric value per year. HISTORICAL YEARS ONLY (YYYY_A or TTM_YYYY).\n"
            "           Do NOT output any projected/estimated years (YYYY_E) — output null for those.\n\n"
            "   EXTRACTION RULES:\n"
            "     STEP 1 — Look for a row explicitly labeled one of (case-insensitive):\n"
            "       a) 'Interest expense'\n"
            "       b) 'Interest expense, net'\n"
            "       c) 'Net interest expense'\n"
            "       d) 'Interest and debt expense'\n"
            "       e) 'Finance costs' or 'Finance charges' (common in Canadian/UK documents)\n"
            "       f) 'Interest charges' or 'Interest on debt'\n"
            "   HARD RULES:\n"
            "     - Extract ONLY if the row is explicitly labeled as one of the above in a structured financial table.\n"
            "     - NEVER extract a row labeled 'Interest and other expense' or 'Interest and other income' —\n"
            "       those are bundled and cannot be separated; output null.\n"
            "     - NEVER derive or calculate interest expense from other line items.\n"
            "     - NEVER extract from charts, dashboards, bullet points, or narrative text.\n"
            "     - NEVER extract from the Balance Sheet (e.g. 'Accrued Interest') — P&L only.\n"
            "     - HISTORICAL YEARS ONLY: If the table shows projected years (YYYY_E), skip them — output null.\n"
            "       Only extract for years with an actual/historical suffix (YYYY_A or TTM_YYYY).\n"
            "     - If a document shows multiple debt instruments separately (Senior debt interest,\n"
            "       Subordinated note interest, etc.) — use ONLY the 'Total interest expense' subtotal row.\n"
            "     - Sign convention: interest expense is typically a positive number (a cost to the business).\n"
            "       If shown in parentheses (e.g. (2,500)) → convert to -2500 and output as-is.\n"
            "     - Convert parentheses to negative: (2,500) = -2500.\n"
            "     - Normalise units to $000s. If CAD, output as-is without USD conversion.\n"
            "     - If not present in the document → output null for all years.\n"
            "     - If a year has no value in an otherwise present row → output null for that year.\n"
            "15. DEPRECIATION EXTRACTION LOGIC:\n"
            "   This is the depreciation (non-cash) charge from the P&L (Excel row 23).\n"
            "   OUTPUT: flat numeric value per year. Extract BOTH historical (YYYY_A/TTM_YYYY) AND projected (YYYY_E) years.\n\n"
            "   EXTRACTION RULES:\n"
            "     STEP 1 — Look for a row explicitly labeled one of (case-insensitive):\n"
            "       a) 'Depreciation'\n"
            "       b) 'Depreciation & Amortization' or 'Depreciation and Amortization' or 'D&A'\n"
            "       c) 'Depreciation of fixed assets' or 'Depreciation of property, plant & equipment'\n"
            "     STEP 2 — If not in the main P&L, also check EBITDA bridge tables where D&A appears\n"
            "       as an add-back line (e.g. 'Operating Income + D&A = EBITDA' format).\n"
            "   HARD RULES:\n"
            "     - Extract ONLY if the row is explicitly labeled as above in a structured financial table.\n"
            "     - If the document shows 'Depreciation' and 'Amortization' on SEPARATE rows, extract\n"
            "       only the 'Depreciation' row — do NOT add them together unless the label says 'D&A'.\n"
            "     - NEVER extract 'Amortization of acquired intangibles' or 'Amortization of intangibles'\n"
            "       as Depreciation — those are post-acquisition accounting items, not operating depreciation.\n"
            "     - NEVER derive or calculate depreciation from other line items.\n"
            "     - NEVER extract from charts, dashboards, bullet points, or narrative text.\n"
            "     - Extract BOTH historical (YYYY_A) AND projected (YYYY_E) years — unlike Interest Expense.\n"
            "     - A value of 0 (zero) is valid — output 0, not null.\n"
            "     - Convert parentheses to negative: (1,200) = -1200.\n"
            "     - Normalise units to $000s. If CAD, output as-is without USD conversion.\n"
            "     - If not present in the document → output null for all years.\n"
            "     - If a year has no value in an otherwise present row → output null for that year.\n"
            "16. CAPEX EXTRACTION LOGIC:\n"
            "   Capital expenditures — cash spent on fixed assets (Excel row 32). Output as NEGATIVE numbers.\n"
            "   Extract BOTH historical (YYYY_A/TTM_YYYY) AND projected (YYYY_E) years.\n\n"
            "   EXTRACTION RULES:\n"
            "     STEP 1 — Look for a row explicitly labeled one of (case-insensitive):\n"
            "       a) 'Total Capex' or 'Total Capital Expenditures' or 'Total CapEx'\n"
            "       b) 'Capital Expenditures' or 'Capex' (standalone summary row)\n"
            "       c) 'Capital Spending' or 'Purchases of property, plant & equipment'\n"
            "   HARD RULES:\n"
            "     - ALWAYS use the 'Total Capex' subtotal row — NEVER sum individual breakdown lines\n"
            "       (e.g. Warehouse Equipment, Building Improvements, Maintenance, Growth, Other).\n"
            "     - If the document shows TWO Total Capex rows (e.g. 'Total Capex' and\n"
            "       'Total Capex (incl. Plant Consolidations)' or similar with one-time items) —\n"
            "       use the FIRST 'Total Capex' row (base recurring capex). NEVER use the inflated\n"
            "       version that includes one-time plant consolidations or non-recurring items.\n"
            "     - OUTPUT AS NEGATIVE: multiply the extracted value by -1 before outputting.\n"
            "       E.g. if the table shows 2,490 → output -2490. If already in parentheses (2,490) → output -2490.\n"
            "       Do NOT double-negate: if source shows (2,490) it is already a cash outflow → output -2490.\n"
            "     - NEVER extract from narrative text, bullet points, or charts.\n"
            "     - NEVER derive capex from fixed asset schedule changes (e.g. delta in gross PP&E).\n"
            "     - Extract BOTH historical (YYYY_A/TTM_YYYY) AND projected (YYYY_E) years.\n"
            "     - Normalise units to $000s. If CAD, output as-is without USD conversion.\n"
            "     - Zero (0) is a valid value — output 0, not null.\n"
            "     - If not present in the document → output null for all years.\n"
            "     - If a year has no value in an otherwise present row → output null for that year.\n"
            "17. WORKING CAPITAL CHANGE EXTRACTION LOGIC:\n"
            "   Year-over-year change in Net Working Capital (Excel row 33).\n"
            "   Sign convention: NWC INCREASE = cash outflow = NEGATIVE. NWC DECREASE = cash inflow = POSITIVE.\n"
            "   Extract BOTH historical (YYYY_A/TTM_YYYY) AND projected (YYYY_E) years.\n\n"
            "   EXTRACTION RULES:\n"
            "     STEP 1 — Look for a row explicitly labeled one of (case-insensitive):\n"
            "       a) 'Change in NWC' or 'Change in Working Capital'\n"
            "       b) 'Changes in Working Capital' or 'Change in Operating Working Capital'\n"
            "       c) '(Growth) / Decline in Net Working Capital' or 'Growth / (Decline) in Net Working Capital'\n"
            "       d) 'Increase / (Decrease) in Working Capital' or '(Increase) / Decrease in Working Capital'\n"
            "       e) 'Net Working Capital Change' or 'Working Capital Change'\n"
            "     STEP 2 — SIGN INTERPRETATION (critical — label wording determines sign):\n"
            "       - If label says 'Change in NWC' or 'Change in Working Capital':\n"
            "           Parentheses value = NWC increased = cash outflow → output as NEGATIVE.\n"
            "           Positive value = NWC decreased = cash inflow → output as POSITIVE.\n"
            "       - If label says '(Growth) / Decline in Net Working Capital':\n"
            "           Positive value = decline in NWC = cash inflow → output as POSITIVE.\n"
            "           Parentheses value = growth in NWC = cash outflow → output as NEGATIVE.\n"
            "           In both cases: parentheses → negative, no parentheses → positive.\n"
            "       - General rule: convert parentheses to negative always. The sign direction is preserved.\n"
            "   HARD RULES:\n"
            "     - Extract ONLY from an explicitly labeled change/movement row — NEVER calculate by subtracting\n"
            "       NWC balances yourself (e.g. do not compute NWC(year2) - NWC(year1)).\n"
            "     - The first year in any table will have no prior period comparison → output null for that year.\n"
            "     - NEVER use 'Net Working Capital' balance rows — only the change/movement row.\n"
            "     - NEVER extract 'Adjusted NWC' balance totals as the change value.\n"
            "     - NEVER extract from narrative text, bullet points, or charts.\n"
            "     - Do NOT derive from components (AR + Inventory + Payables changes).\n"
            "     - Normalise units to $000s. If CAD, output as-is without USD conversion.\n"
            "     - If not present in the document → output null for all years.\n"
            "     - If a year has no value in an otherwise present row → output null for that year."
        )

        user_prompt = (
            f"### CLIENT REQUIREMENTS ###\n{client_requirements}\n\n"
            f"### FINANCIAL CONTEXT (EXTRACTED) ###\n{context_str}\n\n"
            "### OUTPUT FORMAT INSTRUCTION ###\n"
            "Return JSON with this structure (all values in $000s, year keys in YYYY_A / YYYY_E / TTM_YYYY format):\n"
            "{\n"
            "  \"Total_Revenue\": {\"2021_A\": value, \"2022_A\": value, \"2023_A\": value, \"2024_E\": value, ...},\n"
            "  \"Gross_Margin\":   {\"2021_A\": value, \"2022_A\": value, \"2023_A\": value, \"2024_E\": value, ...},\n"
            "  \"SGA\":            {\"2021_A\": value, \"2022_A\": value, \"2023_A\": value, \"2024_E\": value, ...},\n"
            "  \"Operating_Income\": {\n"
            "    \"2021_A\": {\"value\": number_or_null, \"source\": \"extracted|calculated|null\", \"formula\": null_or_string, \"confidence\": \"high|medium|low\"},\n"
            "    \"2022_A\": {\"value\": number_or_null, \"source\": \"...\", \"formula\": null_or_string, \"confidence\": \"...\"},\n"
            "    ...\n"
            "    NOTE: If a year has insufficient data → output null (not an object) for that year.\n"
            "  },\n"
            "  \"Adj_EBITDA\":     {\"2021_A\": value, \"2022_A\": value, \"2023_A\": value, \"2024_E\": value, ...},\n"
            "  \"EBITDA\":         {\"2021_A\": value, \"2022_A\": value, \"2023_A\": value, \"2024_E\": value, ...},\n"
            "  \"Other_Expense\":     {\"2021_A\": value, \"2022_A\": value, \"2023_A\": value, ...},\n"
            "  \"Interest_Expense\":  {\"2021_A\": value, \"2022_A\": value, \"2023_A\": value, ...},\n"
            "  NOTE for Interest_Expense: historical years (YYYY_A / TTM_YYYY) ONLY — null for any YYYY_E.\n"
            "  \"Depreciation\":      {\"2021_A\": value, \"2022_A\": value, \"2023_A\": value, \"2024_E\": value, ...},\n"
            "  NOTE for Depreciation: include BOTH historical (YYYY_A/TTM_YYYY) AND projected (YYYY_E) years.\n"
            "  \"CAPEX\":             {\"2021_A\": value, \"2022_A\": value, \"2023_A\": value, \"2024_E\": value, ...},\n"
            "  NOTE for CAPEX: output as NEGATIVE numbers (e.g. -2490, not 2490). Both historical and projected.\n"
            "  Use 'Total Capex' row only — never breakdown lines. If two Total Capex rows exist, use the first (base, not incl. one-time items).\n"
            "  \"WC_Change\":         {\"2021_A\": value, \"2022_A\": value, \"2023_A\": value, \"2024_E\": value, ...}\n"
            "  NOTE for WC_Change: NWC increase = NEGATIVE (cash outflow), NWC decrease = POSITIVE (cash inflow).\n"
            "  Extract only from explicit change/movement row — NEVER from NWC balance rows or by calculation.\n"
            "  First year in table = null (no prior period). Both historical and projected years.\n"
            "}\n"
            "Include every period found in the document. Use null for any metric not found."
        )

        logging.info(f"Sending extraction request to {self.provider.upper()} ({self.model_name})...")
        try:
            if self.provider in ["openai", "nvidia"]:
                kwargs = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.2 if self.provider == "nvidia" else 0.0,
                }
                
                if self.provider == "openai":
                    kwargs["response_format"] = {"type": "json_object"}
                else:
                    kwargs["top_p"] = 0.7
                    kwargs["max_tokens"] = 4096
                    
                response = self.client.chat.completions.create(**kwargs)
                output = response.choices[0].message.content
                
                # Cleanup potential wrapper blocks if model wasn't strictly forced into json format
                if "```json" in output:
                    output = output.split("```json")[1].split("```")[0]
                elif "```" in output:
                    output = output.split("```")[1].split("```")[0]
                    
                return json.loads(output.strip())
                
            elif self.provider == "anthropic":
                # Anthropic implementation
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4096,
                    temperature=0.0,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": f"{user_prompt}\n\nPlease output only JSON wrapped in ```json ... ``` blocks."}
                    ]
                )
                output_text = response.content[0].text
                
                # Strip markdown ticks for parsing
                if "```json" in output_text:
                    output_text = output_text.split("```json")[1].split("```")[0]
                elif "```" in output_text:
                    output_text = output_text.split("```")[1].split("```")[0]
                    
                return json.loads(output_text.strip())

        except Exception as e:
            logging.error(f"Error during LLM extraction: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description="Parse CIM PDF and extract specific financial fields via LLM.")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the CIM PDF file (e.g., sample_cim.pdf).")
    parser.add_argument(
        "--req", 
        type=str, 
        default=(
            "Extract these financial metrics for ALL available periods found in the document:\n"
            "1) Total_Revenue (also labeled as: net revenue, total net revenue)\n"
            "2) Gross_Margin (also labeled as: contribution margin, gross profit, CM — "
            "the revenue minus direct costs/COGS line, whatever it is called)\n"
            "3) SG&A (also labeled as: selling general & administrative, G&A, operating expenses, "
            "overhead, opex — the expense line sitting BELOW gross margin and ABOVE EBITDA. "
            "Do NOT include COGS or direct costs here)\n"
            "4) Operating_Income — follow the strict 4-step priority logic defined in the system rules (Rule 10).\n"
            "   Per year output: {value, source, formula, confidence}. Output null for years with insufficient data.\n"
            "5) Adj_EBITDA — Adjusted EBITDA per year. Follow the strict priority logic in Rule 11.\n"
            "   Labels to look for (in priority order): 'Adjusted EBITDA', 'Adj. EBITDA', 'Mgmt. Adj. EBITDA',\n"
            "   'Normalized EBITDA', 'PF Adjusted EBITDA' (if no other adjusted variant exists).\n"
            "   If only plain 'EBITDA' exists with no adjustments anywhere in the document, use that.\n"
            "   Output null per year if not found.\n"
            "6) EBITDA — Reported/plain EBITDA per year (BEFORE any adjustments). Follow Rule 12.\n"
            "   Only extract rows explicitly labeled 'Reported EBITDA', 'EBITDA' (no qualifier), or\n"
            "   'EBITDA before adjustments'. NEVER use Adjusted EBITDA for this field.\n"
            "   Output null if only Adjusted EBITDA exists in the document.\n"
            "7) Other_Expense — Other expense / (income) per year. Follow Rule 13.\n"
            "   Labels to look for: 'Other expense/(income)', 'Other (income)/expense', 'Other income/expenses',\n"
            "   'Other income, net', 'Non-operating expense'. Also check EBITDA adjustment tables.\n"
            "   Do NOT extract if label includes 'interest'. Sign: expense=positive, income=negative.\n"
            "   Output null if not explicitly labeled in a financial table.\n"
            "8) Interest_Expense — Interest paid on debt per year. Follow Rule 14.\n"
            "   Labels: 'Interest expense', 'Interest expense, net', 'Net interest expense',\n"
            "   'Finance costs', 'Finance charges', 'Interest charges', 'Interest on debt'.\n"
            "   CRITICAL: Extract HISTORICAL years only (YYYY_A / TTM_YYYY). Output null for YYYY_E.\n"
            "   Do NOT extract if bundled with 'other' (e.g. 'Interest and other expense').\n"
            "   Do NOT extract from balance sheet items like 'Accrued Interest'.\n"
            "   If multiple debt lines present, use 'Total interest expense' subtotal only.\n"
            "   Output null if not explicitly labeled in a P&L financial table.\n"
            "9) Depreciation — Depreciation (non-cash charge) per year. Follow Rule 15.\n"
            "   Labels: 'Depreciation', 'Depreciation & Amortization', 'D&A', 'Depreciation and Amortization'.\n"
            "   Extract BOTH historical (YYYY_A/TTM_YYYY) AND projected (YYYY_E) years.\n"
            "   If 'Depreciation' and 'Amortization' are on separate rows, extract only 'Depreciation'.\n"
            "   NEVER extract 'Amortization of intangibles' for this field.\n"
            "   Zero (0) is a valid value — output 0, not null.\n"
            "   Output null if not explicitly labeled in a financial table.\n"
            "10) CAPEX — Capital expenditures per year. Follow Rule 16.\n"
            "   Labels: 'Total Capex', 'Total Capital Expenditures', 'Capital Expenditures', 'Capex'.\n"
            "   Extract BOTH historical (YYYY_A/TTM_YYYY) AND projected (YYYY_E) years.\n"
            "   OUTPUT AS NEGATIVE: value from table is 2,490 → output -2490.\n"
            "   Use ONLY 'Total Capex' subtotal row — NEVER individual breakdown lines (Maintenance, Growth, etc.).\n"
            "   If two Total Capex rows exist (base vs. incl. one-time items), use the FIRST base row.\n"
            "   NEVER derive from fixed asset schedule changes. NEVER extract from narrative text.\n"
            "   Zero is a valid value — output 0, not null.\n"
            "   Output null if not found in a structured financial table.\n"
            "11) WC_Change — Year-over-year change in Net Working Capital per year. Follow Rule 17.\n"
            "   Labels: 'Change in NWC', 'Change in Working Capital', 'Changes in Working Capital',\n"
            "   '(Growth) / Decline in Net Working Capital', 'Change in Operating Working Capital'.\n"
            "   Sign: NWC increase = NEGATIVE (cash outflow). NWC decrease = POSITIVE (cash inflow).\n"
            "   Parentheses in source always convert to negative. Preserve sign as output.\n"
            "   Extract BOTH historical (YYYY_A/TTM_YYYY) AND projected (YYYY_E) years.\n"
            "   First year in any table = null (no prior period to compare).\n"
            "   NEVER calculate from NWC balance rows. NEVER extract NWC balance as the change.\n"
            "   Output null if no explicit change row found in any financial table.\n\n"
            "Output all numeric values in $000s (thousands). "
            "If currency is CAD, output as-is without USD conversion. "
            "If a metric does not exist in the document, use null."
        ),
        help="Description of exactly what you want extracted from the financials."
    )
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "anthropic", "nvidia"], help="LLM Provider to use (openai, anthropic, or nvidia).")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name (e.g., gpt-4o, claude-3-5-sonnet-20240620, meta/llama-3.3-70b-instruct)")
    parser.add_argument("--api-key", type=str, default=None, help="API key (overrides environment variable).")
    args = parser.parse_args()

    # Authentication — --api-key flag takes priority, then environment variable
    if args.api_key:
        api_key = args.api_key
    else:
        env_var_map = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "nvidia": "NVIDIA_API_KEY"}
        api_key_env = env_var_map[args.provider]
        api_key = os.getenv(api_key_env)

    if not api_key:
        env_var_map = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "nvidia": "NVIDIA_API_KEY"}
        api_key_env = env_var_map[args.provider]
        logging.error(f"Missing API key. Provide it via --api-key or set the {api_key_env} environment variable.")
        exit(1)

    # Execution flow
    try:
        cim = CIMParser(args.pdf)
        cim.parse_pdf()

        relevant_sections = cim.find_financial_sections()
        if not relevant_sections:
            logging.warning("No standard financial keywords found in document. Passing all tables to LLM.")
            # Fallback - just pass tables if we couldn't filter by keywords
            relevant_sections = [t for t in cim.extracted_tables]

        extractor = LLMExtractor(api_key=api_key, provider=args.provider, model_name=args.model)
        result = extractor.extract_fields(relevant_sections, client_requirements=args.req)

        # Output the result
        if result:
            # Map extracted year labels → Excel column labels
            excel_result = extractor.map_to_excel_columns(result)

            print("\n" + "="*50)
            print("EXTRACTED FINANCIAL DATA (raw)")
            print("="*50)
            print(json.dumps(result, indent=4))

            print("\n" + "="*50)
            print("EXCEL COLUMN MAPPING (FY19/FY20/FY21 / Year_1-5)")
            print("="*50)
            print(json.dumps(excel_result, indent=4))

            output_dir = "extracted_results"
            os.makedirs(output_dir, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(args.pdf))[0]

            raw_file = os.path.join(output_dir, f"{base_name}_extracted.json")
            with open(raw_file, 'w') as f:
                json.dump(result, f, indent=4)

            mapped_file = os.path.join(output_dir, f"{base_name}_excel_mapped.json")
            with open(mapped_file, 'w') as f:
                json.dump(excel_result, f, indent=4)

            logging.info(f"Raw JSON saved to {raw_file}")
            logging.info(f"Excel-mapped JSON saved to {mapped_file}")
        else:
            logging.error("The LLM failed to return a valid extraction or threw an error.")
            
    except Exception as e:
        logging.error(f"An unexpected error occurred during execution: {e}")

if __name__ == "__main__":
    main()
