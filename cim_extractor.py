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
                "accounts receivable", "account receivable", "trade receivables",
                "receivables, net", "accounts receivable, net", "net receivables",
                "balance sheet", "current assets", "working capital detail",
                "inventory", "inventories", "inventory, net", "stock",
                "raw materials", "finished goods", "work in progress", "work-in-progress",
                "total inventory",
                # Business overview keywords (for Company_Summary extraction)
                "business overview", "company overview", "executive summary",
                "company description", "business description", "business profile",
                "company profile", "about the company", "about the business",
                "products and services", "product and service", "business model",
                "company history", "key customers", "end markets", "end-markets",
                "investment highlights", "investment overview", "transaction overview",
                "overview", "introduction",
                # Market intelligence keywords (for Market_Intelligence extraction)
                "competitive landscape", "competition", "competitors", "competitive position",
                "competitive overview", "competitive dynamics", "key competitors",
                "market overview", "industry overview", "market size", "market opportunity",
                "total addressable market", "addressable market", "tam", "sam", "som",
                "market growth", "market growth rate", "industry growth", "cagr",
                "market share", "market position", "key players", "major players",
                "industry players", "industry participants", "industry dynamics",
                "market trends", "industry trends", "competitive advantages",
                "barriers to entry", "market leadership", "fragmented market",
                "industry revenue", "industry sales", "total industry", "dealer industry",
                "industry outlook", "industry shipments", "unit shipments",
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
        elif self.provider == "ollama":
            try:
                from openai import OpenAI
                base_url = os.getenv("OLLAMA_BASE_URL", "https://openapi.ollama.ai/v1")
                self.client = OpenAI(base_url=base_url, api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
        else:
            raise ValueError("Unsupported provider. Choose 'openai', 'anthropic', 'nvidia', or 'ollama'.")

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

    @staticmethod
    def _normalize_nulls(result: Dict) -> Dict:
        """
        Convert year-keyed objects where ALL year values are null into scalar null.
        e.g. {"2023_A": null, "2024_A": null} → null
        Leaves Market_Intelligence, Investment_Recommendation, Company_Summary untouched.
        """
        SKIP_KEYS = {"Market_Intelligence", "Investment_Recommendation", "Company_Summary"}
        year_pat = re.compile(r'^\d{4}_(A|E)$|^TTM_\d{4}$|^TTM$')
        normalized = {}
        for key, value in result.items():
            if key in SKIP_KEYS or not isinstance(value, dict):
                normalized[key] = value
            else:
                year_vals = {k: v for k, v in value.items() if year_pat.match(k)}
                if year_vals and all(v is None for v in year_vals.values()):
                    normalized[key] = None
                else:
                    normalized[key] = value
        return normalized

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
            "9. SG&A EXTRACTION RULES:\n"
            "   Extract ONLY if a row is explicitly labeled with one of these names in a structured financial table:\n"
            "   VALID LABELS (extract these):\n"
            "   - 'SG&A', 'SGA', 'Unadjusted SG&A', 'Adjusted SG&A'\n"
            "   - 'Selling, General & Administrative', 'Selling, General and Administrative'\n"
            "   - 'Selling & Marketing', 'Selling Expenses'\n"
            "   - 'G&A', 'General & Administrative', 'General and Administrative'\n"
            "   - 'Overhead' (only if it is a standalone labeled row, not a subtotal)\n"
            "   HARD RULES:\n"
            "   - NEVER extract a row labeled 'Operating Expenses', 'Total Operating Expenses', or 'Opex' as SG&A.\n"
            "     These are calculated subtotal rows (Gross Profit - Operating Income) and will produce wrong values.\n"
            "   - NEVER infer or calculate SG&A from other lines. Only extract if the label is explicit.\n"
            "   - NEVER use 'Any expense line between gross margin and EBITDA' as a rule — that is too vague.\n"
            "   - Do NOT confuse SG&A with COGS or direct costs — those are captured in Gross Margin.\n"
            "   - If no row with a valid SG&A label exists in any financial table → output null for ALL years.\n"
            "   - SG&A is always output as a POSITIVE number (it is a cost/expense).\n"
            "     If the table shows SG&A in parentheses e.g. (13.2) or as a negative → convert to positive.\n"
            "     Example: table shows (13,200) or -13,200 → output 13200.\n"
            "10. ADJ. EBITDA EXTRACTION LOGIC:\n"
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
            "11. EBITDA (REPORTED/PLAIN) EXTRACTION LOGIC:\n"
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
            "12. OTHER EXPENSE / (INCOME) EXTRACTION LOGIC:\n"
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
            "13. INTEREST EXPENSE EXTRACTION LOGIC:\n"
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
            "14. DEPRECIATION EXTRACTION LOGIC:\n"
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
            "15. CAPEX EXTRACTION LOGIC:\n"
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
            "16. WORKING CAPITAL CHANGE EXTRACTION LOGIC:\n"
            "   Year-over-year change in Net Working Capital (Excel row 33).\n"
            "   Sign convention: NWC INCREASE = cash outflow = NEGATIVE. NWC DECREASE = cash inflow = POSITIVE.\n"
            "   Extract BOTH historical (YYYY_A/TTM_YYYY) AND projected (YYYY_E) years where available.\n\n"
            "   STEP 1 — PREFERRED: Look for an explicitly labeled change/movement row (case-insensitive):\n"
            "       a) 'Change in NWC' or 'Change in Working Capital' or 'Changes in Working Capital'\n"
            "       b) 'Change in Operating Working Capital' or 'Net Working Capital Change'\n"
            "       c) '(Growth) / Decline in Net Working Capital' or 'Growth / (Decline) in Net Working Capital'\n"
            "       d) 'Increase / (Decrease) in Working Capital' or '(Increase) / Decrease in Working Capital'\n"
            "       e) 'Working Capital Change'\n"
            "     If found → extract all years from that row directly. Do NOT also apply Step 2.\n\n"
            "   STEP 2 — FALLBACK (only if Step 1 finds nothing): Derive from NWC balance rows.\n"
            "     Condition: a table contains a row labeled 'Net Working Capital', 'Adjusted Net Working Capital',\n"
            "       or 'NWC' with values for 2 or more years.\n"
            "     Method: WC_Change(year N) = NWC(year N) - NWC(year N-1)\n"
            "       - First year in the table → output null (no prior period exists).\n"
            "       - Apply to ALL consecutive year pairs found in the table (historical and projected).\n"
            "       - If only 1 year of NWC balance data exists → output null for all years (cannot derive).\n"
            "     Sign check after derivation:\n"
            "       - Result is POSITIVE → NWC increased → cash outflow → output as NEGATIVE.\n"
            "       - Result is NEGATIVE → NWC decreased → cash inflow → output as POSITIVE.\n"
            "       (Reverse the sign of the raw arithmetic result to match cash flow convention.)\n\n"
            "   SIGN RULES for Step 1 (explicit row):\n"
            "     - Parentheses in source always convert to negative: (9,900) = -9900.\n"
            "     - Labels like '(Growth)/Decline': positive=decline=NWC fell=cash inflow → POSITIVE output.\n"
            "       Parentheses value=growth=NWC rose=cash outflow → NEGATIVE output.\n"
            "     - Labels like 'Change in NWC': parentheses=NWC rose=outflow → NEGATIVE. Positive=NWC fell=inflow → POSITIVE.\n"
            "     - General rule: parentheses → negative, no parentheses → positive. Sign direction is preserved.\n\n"
            "   HARD RULES (both steps):\n"
            "     - NEVER derive from individual component changes (AR, Inventory, Payables separately).\n"
            "     - NEVER extract a single NWC balance value as WC_Change.\n"
            "     - NEVER extract from narrative text, bullet points, or charts.\n"
            "     - Normalise units to $000s. If CAD, output as-is without USD conversion.\n"
            "     - If Step 1 and Step 2 both fail → output null for all years.\n"
            "17. ACCOUNTS RECEIVABLE (AR) EXTRACTION LOGIC:\n"
            "   The AR balance used as ABL collateral in the Sources section (Excel C7).\n"
            "   OUTPUT: a SINGLE numeric value — the most recent actual balance only (NOT a time series).\n\n"
            "   EXTRACTION RULES:\n"
            "     STEP 1 — Find a balance sheet or NWC table containing AR balances across multiple periods.\n"
            "       Look for rows labeled (case-insensitive):\n"
            "         a) 'Accounts Receivable' or 'Account Receivable'\n"
            "         b) 'Accounts Receivable, net' or 'Trade Receivables'\n"
            "         c) 'Receivables, net' or 'Net Receivables' or 'AR'\n"
            "     STEP 2 — Identify the MOST RECENT ACTUAL period in the table:\n"
            "         'Actual' means a year column whose label ends in 'A' or 'a' (e.g. 2024A, FY24A, Sep-24A).\n"
            "         Priority order: TTM_YYYY > highest YYYY_A > YYYY_B (budget, last resort) > YYYY_E (last resort)\n"
            "         *** CRITICAL WARNING: CIM tables often extend 5-8 years of projections beyond the actuals.\n"
            "             The LAST column in the table is almost always a PROJECTED year (E/B/F suffix).\n"
            "             NEVER use it as 'most recent'. Scan left from the right until you find the last 'A' column.\n"
            "             EXAMPLE: columns = 2023A, 2024A, 2025B, 2026E, 2027E, 2028E, 2029E, 2030E\n"
            "                      → most recent actual = 2024A (second column), NOT 2030E (last column). ***\n"
            "         Use the value from that single actual period only.\n"
            "   HARD RULES:\n"
            "     - Output ONE single number — not a dict of years. Format: \"AR\": 6147\n"
            "     - ALWAYS use net value (after allowance for doubtful accounts) if available.\n"
            "       If only gross AR is available, use gross.\n"
            "     - NEVER extract AR from a P&L, income statement, or cash flow adjustments table.\n"
            "       AR must come from a balance sheet or NWC schedule.\n"
            "     - NEVER use AR movement/change values — only the balance (stock) amount.\n"
            "     - NEVER extract from narrative text, bullet points, or charts.\n"
            "     - If multiple balance sheet snapshots exist (e.g. Dec-23A, Dec-24A, Dec-25A),\n"
            "       always pick the most recent one (Dec-25A in this example).\n"
            "     - If only a single snapshot is shown (e.g. 'As of Sep-25'), use that value.\n"
            "     - Normalise to $000s. If CAD, output as-is without USD conversion.\n"
            "     - If AR is not found in any balance sheet or NWC table → output null.\n"
            "18. INVENTORY EXTRACTION LOGIC:\n"
            "   The Inventory balance used as ABL collateral in the Sources section (Excel C8).\n"
            "   OUTPUT: a SINGLE numeric value — the most recent actual balance only (NOT a time series).\n\n"
            "   EXTRACTION RULES:\n"
            "     STEP 1 — Find a balance sheet or NWC table containing inventory balances.\n"
            "       Look for rows labeled (case-insensitive):\n"
            "         a) 'Inventory' or 'Inventories' or 'Inventory, net'\n"
            "         b) 'Stock' (common in UK/Canadian documents)\n"
            "         c) 'Total Inventory' (if broken into components)\n"
            "         d) 'Raw Materials', 'Work in Progress', 'Finished Goods' — BUT only if a\n"
            "            'Total Inventory' subtotal row also exists; use the subtotal, never sum manually.\n"
            "     STEP 2 — Identify the MOST RECENT ACTUAL period in the table:\n"
            "         'Actual' means a year column whose label ends in 'A' or 'a' (e.g. 2024A, FY24A).\n"
            "         Priority order: TTM_YYYY > highest YYYY_A > YYYY_B (budget, last resort) > YYYY_E (last resort)\n"
            "         *** CRITICAL WARNING: CIM tables often extend 5-8 projected years beyond the actuals.\n"
            "             The LAST column in the table is almost always a PROJECTED year (E/B/F suffix).\n"
            "             NEVER use it as 'most recent'. Scan left from the right until you find the last 'A' column.\n"
            "             EXAMPLE: columns = 2023A, 2024A, 2025B, 2026E, ..., 2030E\n"
            "                      → most recent actual = 2024A (second column), NOT 2030E (last column). ***\n"
            "         Use the value from that single actual period only.\n"
            "   HARD RULES:\n"
            "     - Output ONE single number — not a dict of years. Format: \"Inventory\": 13512\n"
            "     - Zero (0) is a valid value — service businesses with no physical goods output 0, not null.\n"
            "     - ALWAYS use net value (after obsolescence/write-down reserve) if shown.\n"
            "       If only gross inventory is available, use gross.\n"
            "     - If inventory is broken into components (Raw Materials / WIP / Finished Goods),\n"
            "       use the 'Total Inventory' subtotal row ONLY. NEVER sum the components yourself.\n"
            "     - NEVER extract inventory movement/change values from a cash flow statement.\n"
            "       Inventory must come from a balance sheet or NWC schedule (a stock/balance, not a flow).\n"
            "     - NEVER extract from narrative text, bullet points, or charts.\n"
            "       Example: 'inventory reduction from $34M in 2022' in narrative → ignore.\n"
            "     - If multiple balance sheet snapshots exist, always pick the most recent period.\n"
            "     - If only a single snapshot is shown (e.g. 'As of Sep-25'), use that value.\n"
            "     - Normalise to $000s. If CAD, output as-is without USD conversion.\n"
            "     - If Inventory is not found in any balance sheet or NWC table → output null.\n"
            "19. MARKET INTELLIGENCE EXTRACTION LOGIC:\n"
            "   Extract competitive landscape and market sizing data from the CIM.\n"
            "   OUTPUT: a structured object with three sub-fields (each can independently be null):\n"
            "     { \"competitors\": [...] or null, \"market_size\": string or null, \"market_growth_rate\": string or null }\n\n"
            "   SUB-FIELD RULES:\n\n"
            "   A) competitors — array of competitor company names:\n"
            "     SOURCE: competitive landscape, competition, market overview sections (narrative text OR comparison tables).\n"
            "     Extract only company/brand names explicitly named as competitors in the document.\n"
            "     EDGE CASES:\n"
            "       - Document names 3–10 competitors → list all named companies as strings.\n"
            "       - Document says 'fragmented market with no dominant competitor' → output null (no names given).\n"
            "       - Document lists competitors only in a comparison table → still extract the names.\n"
            "       - Company itself appears in a competitor table → exclude it from the list.\n"
            "       - Document mentions competitors only in passing ('competes with large public companies') → null.\n"
            "       - Competitor described by category only ('large distributors') with no name → skip it.\n"
            "       - No competitive section present at all → null.\n"
            "     MAX 12 competitors. Output as array of strings, e.g. [\"Company A\", \"Company B\"].\n"
            "     If no named competitors found → output null (not an empty array).\n\n"
            "   B) market_size — descriptive string for the overall industry/market size:\n"
            "     SOURCE: market overview, industry overview, investment highlights, competitive landscape sections.\n"
            "     Accept ANY of these labels as a market size indicator (case-insensitive):\n"
            "       - 'Total Addressable Market', 'TAM', 'Addressable Market', 'Market Size'\n"
            "       - 'Industry Revenue', 'Industry Sales', 'Total Industry Revenue', 'Total Industry Sales'\n"
            "       - '[Industry Name] Market', e.g. 'RV Market', 'U.S. Motorhome Market', 'North American Market'\n"
            "       - '[Industry Name] Dealer Industry Revenue' or '[Industry Name] Industry Revenue Outlook'\n"
            "       - 'Total Market', 'Overall Market', 'Global Market', 'North American Market'\n"
            "       - Any figure describing the size of the industry the company operates in\n"
            "     Extract the figure with units and year if stated (e.g. '$48.9 billion USD (2024)').\n"
            "     EDGE CASES:\n"
            "       - Given as a range ('$3–5 billion') → preserve as-is: '$3–5 billion'.\n"
            "       - Given for multiple segments → extract total/combined figure if stated; if not, describe\n"
            "         the segments (e.g. 'Segment A: $1.2B; Segment B: $0.8B').\n"
            "       - Market size in CAD or another currency → preserve with currency symbol.\n"
            "       - Figure cited from a third-party source ('per XYZ Research, $6B market') → extract\n"
            "         the figure only, omit the source name.\n"
            "       - Industry revenue given for current year AND projected years → use most recent actual/stated year.\n"
            "       - Market size stated only as 'large' or 'growing' with no numeric figure → null.\n"
            "       - No industry/market size figure anywhere in the document → null.\n"
            "     Do NOT convert or reformat the number — preserve units exactly as in the document.\n\n"
            "   C) market_growth_rate — descriptive string for market growth:\n"
            "     SOURCE: same market/industry overview sections.\n"
            "     Extract the stated CAGR or growth rate with time horizon if given (e.g. '~8% CAGR (2024–2028)').\n"
            "     EDGE CASES:\n"
            "       - Given as a range ('6–9% CAGR') → preserve as-is.\n"
            "       - Stated as historical only ('grew 7% in 2023') → extract with year.\n"
            "       - Only qualitative ('fast-growing', 'double-digit growth') with no number → null.\n"
            "       - Multiple growth rates for different segments → take the overall market rate if stated;\n"
            "         if only segment rates given, use largest or most relevant segment, note it.\n"
            "       - No growth figure anywhere → null.\n\n"
            "   GLOBAL HARD RULES for Market_Intelligence:\n"
            "     - NEVER hallucinate or infer competitors, market sizes, or growth rates not in the document.\n"
            "     - NEVER extract from financial tables (P&L, balance sheet) — narrative and overview sections only.\n"
            "     - NEVER include financial metrics (revenue, EBITDA, margins) inside Market_Intelligence.\n"
            "     - If the CIM has no market/competitive overview section at all → output null for the entire field.\n"
            "     - Output the full structured object even if only one sub-field has data; set others to null.\n"
            "20. 1X ADJUSTMENTS EXTRACTION LOGIC:\n"
            "   These are non-recurring / one-time add-backs in the EBITDA bridge table that reconcile\n"
            "   Reported EBITDA to Adjusted EBITDA. Extract the TOTAL adjustment amount per year.\n"
            "   OUTPUT: flat numeric value per year (positive = net add-back). Historical years ONLY (YYYY_A / TTM_YYYY).\n"
            "   Do NOT extract projected (YYYY_E) years — projected adjustments are always $0 by convention.\n\n"
            "   STEP 1 — DIRECT TOTAL ROW (preferred):\n"
            "     Scan EBITDA bridge / adjustment schedule tables for a row explicitly labeled:\n"
            "     'Total Adjustments', 'Total Management Adjustments', 'Total normalizations',\n"
            "     'Net Adjustments', 'Total Add-backs', 'Total 1x Items'.\n"
            "     If found → extract the per-year values from that row.\n\n"
            "   STEP 2 — DERIVE: Adj_EBITDA minus EBITDA (fallback only):\n"
            "     Only attempt if Step 1 fails (no explicit total row found).\n"
            "     Conditions — BOTH must be present in the SAME table or from already-extracted values:\n"
            "       a) Adj_EBITDA value for the year\n"
            "       b) Plain EBITDA (Reported EBITDA) value for the year\n"
            "     Formula: Onex_Adjustments = Adj_EBITDA - EBITDA\n"
            "     Only compute for years where BOTH values are non-null.\n\n"
            "   STEP 3 — NULL:\n"
            "     If neither Step 1 nor Step 2 succeeds → output null.\n\n"
            "   HARD RULES:\n"
            "     - NEVER use individual adjustment line items (e.g. 'Management Fees', 'Legal') — use the TOTAL only.\n"
            "     - NEVER extract Pro Forma / synergy adjustment tiers — Management adjustments only.\n"
            "     - Sign: positive = net add-back (increases EBITDA). If total row shows parentheses → negative.\n"
            "     - HISTORICAL ONLY: do NOT output YYYY_E values (projected 1x = $0, not meaningful).\n"
            "     - If projected years show $0 in the total row, output null for those years (not 0).\n"
            "     - NEVER extract from P&L rows labeled 'Other Expense' or 'Other Income' unless in bridge table.\n"
            "     - NEVER mix adjustment tiers (e.g. do not add Management + PF + Synergy together).\n"
            "21. COMPANY SUMMARY EXTRACTION LOGIC:\n"
            "   A concise plain-English description of what the company does.\n"
            "   OUTPUT: a single string. Target length: 200–250 words. Must be in English.\n\n"
            "   EXTRACTION RULES:\n"
            "     STEP 1 — Read the business overview, executive summary, company profile, or\n"
            "       investment highlights section (typically the first 5–15 pages of the CIM).\n"
            "     STEP 2 — Write a factual, neutral summary covering:\n"
            "       a) What the company does — its core business and products/services\n"
            "       b) Key end markets or customer segments it serves\n"
            "       c) Business model (how it makes money — e.g. recurring contracts, direct sales, etc.)\n"
            "       d) Geographic presence if mentioned\n"
            "       e) Any standout competitive advantages or brief company history if stated\n"
            "   HARD RULES:\n"
            "     - Write as a neutral third-party analyst — no promotional language, no exclamation marks.\n"
            "     - Do NOT quote raw CIM text verbatim — paraphrase into clean analytical prose.\n"
            "     - Do NOT include any financial figures (revenue, EBITDA, margins, etc.) in this summary.\n"
            "     - Do NOT reference the CIM document itself ('this document states...', 'per the CIM...').\n"
            "     - Keep it strictly 200–250 words — do not go below 180 or above 270.\n"
            "     - If no business overview or company description section is present → output null.\n"
            "     - Output format: plain string, no bullet points, no markdown headers."
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
            "  \"WC_Change\":         {\"2021_A\": value, \"2022_A\": value, \"2023_A\": value, \"2024_E\": value, ...},\n"
            "  NOTE for WC_Change: NWC increase = NEGATIVE (cash outflow), NWC decrease = POSITIVE (cash inflow).\n"
            "  STEP 1: extract from explicit labeled change row if present.\n"
            "  STEP 2 fallback: if no change row, derive from NWC balance rows (need 2+ years; first year = null).\n"
            "  Derivation sign: WC_Change = NWC(N) - NWC(N-1), then REVERSE sign for cash flow convention.\n"
            "  Both historical and projected years. If only 1 NWC balance year exists → null.\n"
            "  \"Onex_Adjustments\": {\"2021_A\": value, \"2022_A\": value, \"2023_A\": value, ...},\n"
            "  NOTE for Onex_Adjustments: HISTORICAL years (YYYY_A / TTM_YYYY) ONLY — never output YYYY_E.\n"
            "  STEP 1: find explicit total row ('Total Adjustments', 'Total normalizations', 'Total Management Adjustments').\n"
            "  STEP 2 fallback: derive as Adj_EBITDA - EBITDA per year (only if both non-null).\n"
            "  Positive = net add-back. Projected $0 rows → output null, not 0. Null if no bridge table exists.\n"
            "  \"AR\": single_number_or_null,\n"
            "  NOTE for AR: ONE single value — last 'A'-suffixed actual year only (e.g. 2024A not 2030E). NOT a time series.\n"
            "  NEVER use projected (E/B/F) columns. Table may extend to 2030E — use only the last actual column.\n"
            "  \"Inventory\": single_number_or_null,\n"
            "  NOTE for Inventory: ONE single value — last 'A'-suffixed actual year only (e.g. 2024A not 2030E). NOT a time series.\n"
            "  NEVER use projected (E/B/F) columns. Table may extend to 2030E — use only the last actual column.\n"
            "  Zero (0) is valid for service businesses. Use Total Inventory subtotal if broken into components.\n"
            "  NEVER extract from narrative text or cash flow movements — balance sheet/NWC table only.\n"
            "  \"Market_Intelligence\": {\n"
            "    \"competitors\": [\"Name A\", \"Name B\", ...] or null,\n"
            "    \"market_size\": \"descriptive string with units and year\" or null,\n"
            "    \"market_growth_rate\": \"descriptive string with CAGR and horizon\" or null\n"
            "  },\n"
            "  NOTE for Market_Intelligence: extract from competitive landscape / market overview narrative sections.\n"
            "  competitors = named company strings only (max 12) — null if no names given or section absent.\n"
            "  market_size = preserve original figure + units + year (e.g. '$4.2 billion (2024)') — null if not stated.\n"
            "  market_growth_rate = preserve CAGR/rate + horizon (e.g. '~8% CAGR 2024-2028') — null if qualitative only.\n"
            "  If entire market/competitive section absent → set the full field to null (not an object).\n"
            "  \"Company_Summary\": \"string of 200-250 words or null\"\n"
            "  NOTE for Company_Summary: plain English prose, 200-250 words, no financial figures, no bullet points.\n"
            "  Describe: what the company does, end markets, business model, geography, competitive position.\n"
            "  Source: business overview / executive summary / company profile section of the CIM (NOT financial tables).\n"
            "  Output null if no overview section is present.\n"
            "}\n"
            "Include every period found in the document. Use null for any metric not found."
        )

        logging.info(f"Sending extraction request to {self.provider.upper()} ({self.model_name})...")
        try:
            if self.provider in ["openai", "nvidia", "ollama"]:
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
                elif self.provider == "ollama":
                    kwargs["temperature"] = 0.1
                    kwargs["max_tokens"] = 8192
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

                return self._normalize_nulls(json.loads(output.strip()))
                
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

                return self._normalize_nulls(json.loads(output_text.strip()))

        except Exception as e:
            logging.error(f"Error during LLM extraction: {e}")
            return None

    def generate_investment_recommendation(self, extracted_data: Dict, deal_value: float) -> Optional[Dict]:
        """
        Makes a second LLM call using the already-extracted data + deal value
        to produce a structured M&A investment recommendation.
        """
        system_prompt = (
            "You are a senior M&A analyst at a private equity firm. "
            "You have been given extracted financial and market data from a CIM (Confidential Information Memorandum) "
            "and an enterprise value (deal price) the client is considering paying. "
            "Your job is to give a clear, honest investment recommendation.\n\n"
            "CRITICAL RULES:\n"
            "1. Base your recommendation on the data provided — do NOT assume or invent figures not given.\n"
            "2. If a key metric is null/missing, reason from what IS available and explicitly flag the gap.\n"
            "3. Never give a false sense of certainty — if data is sparse, say so and reduce confidence.\n"
            "4. Be direct and concise — this is for a sophisticated PE client, not a retail investor.\n"
            "5. Output ONE valid JSON object and nothing else.\n\n"
            "VERDICT OPTIONS: 'Strong Buy', 'Buy', 'Caution', 'Pass', 'Insufficient Data'\n"
            "CONFIDENCE OPTIONS: 'High', 'Medium', 'Low'\n\n"
            "VERDICT GUIDANCE (EV/EBITDA based — adjust up/down for data quality and trends):\n"
            "  < 6x  → lean Strong Buy (if revenue growing and margins healthy)\n"
            "  6–9x  → Buy\n"
            "  9–12x → Caution\n"
            "  > 12x → Pass (unless exceptional growth justifies premium)\n"
            "  EBITDA null AND Revenue null → Insufficient Data\n\n"
            "CONFIDENCE GUIDANCE:\n"
            "  High   — EBITDA + Revenue + 3 or more supporting fields present\n"
            "  Medium — EBITDA or Revenue present + at least 1 supporting field\n"
            "  Low    — Only 1 key metric present, rest null\n\n"
            "FREE CASH FLOW NOTE: If CAPEX is available, estimate FCF = EBITDA + CAPEX (CAPEX is negative). "
            "High capex drain (|CAPEX| > 50% of EBITDA) is a risk factor. "
            "Persistent negative WC_Change (growing NWC) also reduces real free cash flow.\n\n"
            "MARKET CONTEXT: Use Market_Intelligence if present to assess tailwinds/headwinds. "
            "Growing market (positive CAGR) = positive signal. Fragmented competitive market = pricing risk.\n\n"
            "OUTPUT FORMAT — return exactly this JSON structure:\n"
            "{\n"
            "  \"verdict\": \"Strong Buy|Buy|Caution|Pass|Insufficient Data\",\n"
            "  \"confidence\": \"High|Medium|Low\",\n"
            "  \"ev_ebitda_multiple\": number_or_null,\n"
            "  \"ev_revenue_multiple\": number_or_null,\n"
            "  \"key_positives\": [\"concise point\", \"concise point\", ...],\n"
            "  \"key_risks\": [\"concise point\", \"concise point\", ...],\n"
            "  \"data_gaps\": [\"field name\", ...],\n"
            "  \"rationale\": \"2-3 sentence plain-English summary of the recommendation\"\n"
            "}\n"
            "key_positives and key_risks: 2-4 points each, short and specific.\n"
            "data_gaps: list only fields that were null/missing AND were relevant to the analysis.\n"
            "ev_ebitda_multiple: round to 1 decimal. Use most recent actual EBITDA; if null use first projected.\n"
            "ev_revenue_multiple: round to 2 decimals. Same priority.\n"
            "If both EBITDA and Revenue are null → verdict must be 'Insufficient Data'."
        )

        user_prompt = (
            f"DEAL VALUE (Enterprise Value being considered): ${deal_value:,.0f} (in USD, as entered by client)\n"
            f"NOTE: All financial values below are in $000s (thousands).\n\n"
            f"EXTRACTED FINANCIAL DATA:\n{json.dumps(extracted_data, indent=2)}\n\n"
            "Using the data above and the deal value provided, generate the investment recommendation JSON."
        )

        logging.info(f"Generating investment recommendation (deal value: ${deal_value:,.0f})...")
        try:
            if self.provider in ["openai", "nvidia", "ollama"]:
                kwargs = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.1,
                }
                if self.provider == "openai":
                    kwargs["response_format"] = {"type": "json_object"}
                elif self.provider == "nvidia":
                    kwargs["top_p"] = 0.7
                    kwargs["max_tokens"] = 2048
                elif self.provider == "ollama":
                    kwargs["max_tokens"] = 2048

                response = self.client.chat.completions.create(**kwargs)
                output = response.choices[0].message.content
                if "```json" in output:
                    output = output.split("```json")[1].split("```")[0]
                elif "```" in output:
                    output = output.split("```")[1].split("```")[0]
                return json.loads(output.strip())

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=2048,
                    temperature=0.1,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": f"{user_prompt}\n\nOutput only JSON wrapped in ```json ... ``` blocks."}
                    ]
                )
                output_text = response.content[0].text
                if "```json" in output_text:
                    output_text = output_text.split("```json")[1].split("```")[0]
                elif "```" in output_text:
                    output_text = output_text.split("```")[1].split("```")[0]
                return json.loads(output_text.strip())

        except Exception as e:
            logging.error(f"Error generating investment recommendation: {e}")
            return None

DEFAULT_CLIENT_REQ = (
            "Extract these financial metrics for ALL available periods found in the document:\n"
            "1) Total_Revenue (also labeled as: net revenue, total net revenue)\n"
            "2) Gross_Margin (also labeled as: contribution margin, gross profit, CM — "
            "the revenue minus direct costs/COGS line, whatever it is called)\n"
            "3) SGA — Selling, General & Administrative expenses. Follow Rule 9 strictly.\n"
            "   VALID labels: 'SG&A', 'SGA', 'Unadjusted SG&A', 'Selling, General & Administrative',\n"
            "   'G&A', 'General & Administrative', 'Selling & Marketing', 'Overhead' (standalone only).\n"
            "   NEVER extract rows labeled 'Operating Expenses' or 'Total Operating Expenses' for this field.\n"
            "   NEVER calculate or infer SG&A — only extract if explicitly labeled in a financial table.\n"
            "   Output as POSITIVE number always — if table shows parentheses or negative, convert to positive.\n"
            "   Output null for ALL years if no valid SG&A label is found in the document.\n"
            "4) Adj_EBITDA — Adjusted EBITDA per year. Follow the strict priority logic in Rule 10.\n"
            "   Labels to look for (in priority order): 'Adjusted EBITDA', 'Adj. EBITDA', 'Mgmt. Adj. EBITDA',\n"
            "   'Normalized EBITDA', 'PF Adjusted EBITDA' (if no other adjusted variant exists).\n"
            "   If only plain 'EBITDA' exists with no adjustments anywhere in the document, use that.\n"
            "   Output null per year if not found.\n"
            "5) EBITDA — Reported/plain EBITDA per year (BEFORE any adjustments). Follow Rule 11.\n"
            "   Only extract rows explicitly labeled 'Reported EBITDA', 'EBITDA' (no qualifier), or\n"
            "   'EBITDA before adjustments'. NEVER use Adjusted EBITDA for this field.\n"
            "   Output null if only Adjusted EBITDA exists in the document.\n"
            "6) Other_Expense — Other expense / (income) per year. Follow Rule 12.\n"
            "   Labels to look for: 'Other expense/(income)', 'Other (income)/expense', 'Other income/expenses',\n"
            "   'Other income, net', 'Non-operating expense'. Also check EBITDA adjustment tables.\n"
            "   Do NOT extract if label includes 'interest'. Sign: expense=positive, income=negative.\n"
            "   Output null if not explicitly labeled in a financial table.\n"
            "7) Interest_Expense — Interest paid on debt per year. Follow Rule 13.\n"
            "   Labels: 'Interest expense', 'Interest expense, net', 'Net interest expense',\n"
            "   'Finance costs', 'Finance charges', 'Interest charges', 'Interest on debt'.\n"
            "   CRITICAL: Extract HISTORICAL years only (YYYY_A / TTM_YYYY). Output null for YYYY_E.\n"
            "   Do NOT extract if bundled with 'other' (e.g. 'Interest and other expense').\n"
            "   Do NOT extract from balance sheet items like 'Accrued Interest'.\n"
            "   If multiple debt lines present, use 'Total interest expense' subtotal only.\n"
            "   Output null if not explicitly labeled in a P&L financial table.\n"
            "8) Depreciation — Depreciation (non-cash charge) per year. Follow Rule 14.\n"
            "   Labels: 'Depreciation', 'Depreciation & Amortization', 'D&A', 'Depreciation and Amortization'.\n"
            "   Extract BOTH historical (YYYY_A/TTM_YYYY) AND projected (YYYY_E) years.\n"
            "   If 'Depreciation' and 'Amortization' are on separate rows, extract only 'Depreciation'.\n"
            "   NEVER extract 'Amortization of intangibles' for this field.\n"
            "   Zero (0) is a valid value — output 0, not null.\n"
            "   Output null if not explicitly labeled in a financial table.\n"
            "9) CAPEX — Capital expenditures per year. Follow Rule 15.\n"
            "   Labels: 'Total Capex', 'Total Capital Expenditures', 'Capital Expenditures', 'Capex'.\n"
            "   Extract BOTH historical (YYYY_A/TTM_YYYY) AND projected (YYYY_E) years.\n"
            "   OUTPUT AS NEGATIVE: value from table is 2,490 → output -2490.\n"
            "   Use ONLY 'Total Capex' subtotal row — NEVER individual breakdown lines (Maintenance, Growth, etc.).\n"
            "   If two Total Capex rows exist (base vs. incl. one-time items), use the FIRST base row.\n"
            "   NEVER derive from fixed asset schedule changes. NEVER extract from narrative text.\n"
            "   Zero is a valid value — output 0, not null.\n"
            "   Output null if not found in a structured financial table.\n"
            "10) WC_Change — Year-over-year change in Net Working Capital. Follow Rule 16 (two-step logic).\n"
            "   STEP 1 (preferred): extract from explicit labeled change row if present.\n"
            "     Labels: 'Change in NWC', 'Change in Working Capital', 'Changes in Working Capital',\n"
            "     '(Growth) / Decline in Net Working Capital', 'Change in Operating Working Capital'.\n"
            "   STEP 2 (fallback): if no change row exists, derive from NWC balance rows:\n"
            "     WC_Change(year N) = NWC(year N) - NWC(year N-1), then REVERSE sign.\n"
            "     First year in table = null. Requires 2+ NWC balance years. If only 1 year → null.\n"
            "   Sign convention always: NWC increase = NEGATIVE. NWC decrease = POSITIVE.\n"
            "   Extract BOTH historical (YYYY_A/TTM_YYYY) AND projected (YYYY_E) years.\n"
            "   NEVER derive from individual components (AR, Inventory, Payables separately).\n"
            "   Output null if both steps fail (no change row and fewer than 2 NWC balance years).\n"
            "11) AR — Accounts Receivable balance for ABL collateral (Sources section C7). Follow Rule 17.\n"
            "   Labels: 'Accounts Receivable', 'Account Receivable, net', 'Trade Receivables', 'Receivables, net'.\n"
            "   OUTPUT: ONE single number (not a time series) — the most recent ACTUAL balance only.\n"
            "   'Actual' = last year column ending in 'A' or 'a' (e.g. 2024A). NEVER use E/B/F projected columns.\n"
            "   WARNING: If table spans e.g. 2023A–2030E, the most recent actual is 2024A, NOT the last column.\n"
            "   Period priority: TTM > latest YYYY_A > YYYY_B > YYYY_E (only if no actual exists).\n"
            "   Source: balance sheet or NWC schedule table ONLY — never from P&L or cash flow table.\n"
            "   Use net value (after allowance) if available. Normalise to $000s.\n"
            "   Output null if AR not found in any balance sheet or NWC table.\n"
            "12) Inventory — Inventory balance for ABL collateral (Sources section C8). Follow Rule 18.\n"
            "   Labels: 'Inventory', 'Inventories', 'Inventory, net', 'Stock', 'Total Inventory'.\n"
            "   OUTPUT: ONE single number — the most recent ACTUAL balance only (not a time series).\n"
            "   'Actual' = last year column ending in 'A' or 'a'. NEVER use E/B/F projected columns.\n"
            "   WARNING: If table spans e.g. 2023A–2030E, the most recent actual is 2024A, NOT the last column.\n"
            "   Period priority: TTM > latest YYYY_A > YYYY_B > YYYY_E (only if no actual exists).\n"
            "   Source: balance sheet or NWC schedule table ONLY — never from P&L or cash flow table.\n"
            "   Zero (0) is valid — service businesses with no physical goods output 0, not null.\n"
            "   If broken into components (Raw Materials/WIP/Finished Goods), use 'Total Inventory' subtotal only.\n"
            "   Use net value (after obsolescence reserve) if shown. Normalise to $000s.\n"
            "   NEVER extract from narrative text (e.g. 'inventory reduced from $34M') — table values only.\n"
            "   Output null if Inventory not found in any balance sheet or NWC table.\n"
            "13) Market_Intelligence — Competitive landscape and market sizing. Follow Rule 19.\n"
            "   OUTPUT: structured object with exactly these three sub-fields:\n"
            "     competitors: array of named competitor company strings, or null if none explicitly named.\n"
            "     market_size: string with figure + units + year if stated (e.g. '$48.9 billion USD (2024)'), or null.\n"
            "       Accept any label: TAM, Addressable Market, Industry Revenue, Total Industry Sales,\n"
            "       '[Industry] Market', '[Industry] Dealer Industry Revenue', Total Market, etc.\n"
            "     market_growth_rate: string with CAGR/growth rate + time horizon if stated, or null.\n"
            "   SOURCE: competitive landscape, market overview, industry overview, investment highlights\n"
            "   sections — narrative text AND comparison tables. NOT financial P&L/balance sheet tables.\n"
            "   CRITICAL EDGE CASES:\n"
            "     - Only extract named competitors — 'large players' with no name → skip.\n"
            "     - Fragmented market with no named players → competitors: null.\n"
            "     - Market size as range → preserve as string. Multi-segment → total if stated.\n"
            "     - Qualitative growth only ('fast-growing') → market_growth_rate: null.\n"
            "     - If entire market/competitive section is absent → output null for the whole field.\n"
            "     - NEVER invent data not in the document.\n"
            "14) Onex_Adjustments — Total non-recurring / 1x adjustment amount per year. Follow Rule 20.\n"
            "   OUTPUT: flat numeric value per year (positive = net add-back). Historical years ONLY (YYYY_A / TTM_YYYY).\n"
            "   STEP 1: find an explicit total row in the EBITDA bridge table:\n"
            "     Labels: 'Total Adjustments', 'Total normalizations', 'Total Management Adjustments',\n"
            "     'Net Adjustments', 'Total Add-backs'.\n"
            "   STEP 2 fallback: derive as Adj_EBITDA - EBITDA for each year (only if both non-null).\n"
            "   NEVER use individual line items — use the total only.\n"
            "   NEVER include PF or synergy adjustment tiers — Management adjustments only.\n"
            "   Projected years (YYYY_E): do NOT output even if $0 appears — output null.\n"
            "   Output null if no EBITDA bridge table exists in the document.\n"
            "15) Company_Summary — A concise analyst-written description of the company. Follow Rule 21.\n"
            "   Source: executive summary, business overview, company profile, or investment highlights\n"
            "   sections (typically the first 5–15 pages of the CIM).\n"
            "   Length: 200–250 words. Plain English prose — no bullet points, no financial figures.\n"
            "   Neutral tone: describe what the company does, its end markets, business model,\n"
            "   geographic presence, and competitive position.\n"
            "   Output null if no business overview section is present in the document.\n\n"
            "Output all numeric values in $000s (thousands). "
            "If currency is CAD, output as-is without USD conversion. "
            "If a metric does not exist in the document, use null."
)

def main():
    parser = argparse.ArgumentParser(description="Parse CIM PDF and extract specific financial fields via LLM.")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the CIM PDF file (e.g., sample_cim.pdf).")
    parser.add_argument(
        "--req",
        type=str,
        default=DEFAULT_CLIENT_REQ,
        help="Description of exactly what you want extracted from the financials."
    )
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "anthropic", "nvidia", "ollama"], help="LLM Provider to use (openai, anthropic, nvidia, or ollama).")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name (e.g., gpt-4o, claude-3-5-sonnet-20240620, meta/llama-3.3-70b-instruct)")
    parser.add_argument("--api-key", type=str, default=None, help="API key (overrides environment variable).")
    parser.add_argument("--deal-value", type=float, default=None, help="Enterprise value / deal price being considered (in USD). If provided, generates an investment recommendation.")
    args = parser.parse_args()

    # Authentication — --api-key flag takes priority, then environment variable
    if args.api_key:
        api_key = args.api_key
    else:
        env_var_map = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "nvidia": "NVIDIA_API_KEY", "ollama": "OLLAMA_API_KEY"}
        api_key_env = env_var_map[args.provider]
        api_key = os.getenv(api_key_env)

    if not api_key:
        env_var_map = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "nvidia": "NVIDIA_API_KEY", "ollama": "OLLAMA_API_KEY"}
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

        # Generate investment recommendation if deal value provided
        if result and args.deal_value is not None:
            recommendation = extractor.generate_investment_recommendation(result, args.deal_value)
            if recommendation:
                # Insert recommendation between Market_Intelligence and Company_Summary
                ordered = {}
                for key in result:
                    if key == "Company_Summary":
                        ordered["Investment_Recommendation"] = recommendation
                    ordered[key] = result[key]
                # If Company_Summary was not present, append at end before it
                if "Investment_Recommendation" not in ordered:
                    ordered["Investment_Recommendation"] = recommendation
                result = ordered

        # Output the result
        if result:
            print("\n" + "="*50)
            print("EXTRACTED FINANCIAL DATA (raw)")
            print("="*50)
            print(json.dumps(result, indent=4))

            output_dir = "extracted_results"
            os.makedirs(output_dir, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(args.pdf))[0]

            raw_file = os.path.join(output_dir, f"{base_name}_extracted.json")
            with open(raw_file, 'w') as f:
                json.dump(result, f, indent=4)

            logging.info(f"Raw JSON saved to {raw_file}")
        else:
            logging.error("The LLM failed to return a valid extraction or threw an error.")
            
    except Exception as e:
        logging.error(f"An unexpected error occurred during execution: {e}")

if __name__ == "__main__":
    main()
