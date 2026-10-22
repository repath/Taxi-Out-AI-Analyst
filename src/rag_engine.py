"""
Lightweight, Precise RAG Engine for AI Analyst.
Designed for JNTU Academic Coursework:
- Sentence-level / focused paragraph extraction to answer queries directly.
- Limits scope strictly to what was asked (no dump of irrelevant surrounding text).
- Direct fast-path for common factual queries (record counts, model scores, cleaning metrics).
- Zero mandatory API costs or credits wasted.
"""

import os
import re
import json
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.analytics import build_monthly_statistics


def load_markdown_documents(base_dir: str) -> List[Dict[str, str]]:
    """
    Scans the repository for core project markdown documentation.
    """
    target_docs = [
        os.path.join("docs", "ai_analyst_knowledge.md"),
        "Project_Report.md",
        "README.md",
        "PARAMETERS.md",
        "IMPROVEMENTS.md",
        os.path.join("docs", "architecture.md"),
        os.path.join("docs", "data_dictionary.md"),
        os.path.join("docs", "user_guide.md"),
        os.path.join("output", "pdf", "2606MGAIL130_Abstract.md"),
    ]

    documents = []
    for rel_path in target_docs:
        full_path = os.path.join(base_dir, rel_path)
        if os.path.exists(full_path):
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    documents.append({
                        "file_path": rel_path,
                        "file_name": os.path.basename(rel_path),
                        "content": content
                    })
            except Exception as e:
                print(f"Warning: could not read {full_path}: {e}")

    return documents


def chunk_markdown(doc: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Splits markdown documents into granular, topic-focused snippets (paragraphs, table blocks).
    """
    content = doc["content"]
    file_path = doc["file_path"]
    
    # Split by headings
    sections = re.split(r'(?m)(?=^#{1,4}\s+)', content)
    chunks = []
    
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        
        lines = sec.split("\n")
        title = lines[0].lstrip("#").strip() if lines[0].startswith("#") else doc["file_name"]
        
        # Split section into smaller paragraphs / sub-elements
        paragraphs = sec.split("\n\n")
        for p in paragraphs:
            p_clean = p.strip()
            # Ignore purely heading-only paragraphs or extremely short lines.
            # Long headings must also be excluded; otherwise a title such as
            # "Data cleaning and quality controls" can be returned as the answer.
            if (
                not p_clean
                or len(p_clean) < 30
                or re.fullmatch(r"#{1,6}\s+.+", p_clean)
            ):
                continue
            
            chunks.append({
                "file_path": file_path,
                "section": title,
                "text": p_clean
            })

    return chunks

def load_xlsx_sheet_records(file_path: str, sheet_name: str) -> List[Dict[str, str]]:
    """Reads a simple worksheet table using only Python's standard library."""
    spreadsheet_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    office_rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    package_rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"

    def column_index(cell_reference: str) -> int:
        letters = re.match(r"[A-Z]+", cell_reference.upper())
        index = 0
        for letter in letters.group(0) if letters else "A":
            index = index * 26 + ord(letter) - ord("A") + 1
        return index - 1

    with zipfile.ZipFile(file_path) as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall(f"{{{spreadsheet_ns}}}si"):
                shared_strings.append("".join(node.text or "" for node in item.iter(f"{{{spreadsheet_ns}}}t")))

        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        relationship_id = None
        for sheet in workbook_root.findall(f".//{{{spreadsheet_ns}}}sheet"):
            if sheet.get("name") == sheet_name:
                relationship_id = sheet.get(f"{{{office_rel_ns}}}id")
                break
        if not relationship_id:
            return []

        relationships_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = None
        for relationship in relationships_root.findall(f"{{{package_rel_ns}}}Relationship"):
            if relationship.get("Id") == relationship_id:
                target = relationship.get("Target")
                break
        if not target:
            return []

        worksheet_path = target.lstrip("/")
        if not worksheet_path.startswith("xl/"):
            worksheet_path = f"xl/{worksheet_path}"
        worksheet_root = ET.fromstring(archive.read(worksheet_path))
        rows = []
        for row in worksheet_root.findall(f".//{{{spreadsheet_ns}}}row"):
            values = {}
            for cell in row.findall(f"{{{spreadsheet_ns}}}c"):
                index = column_index(cell.get("r", "A1"))
                cell_type = cell.get("t")
                if cell_type == "inlineStr":
                    value = "".join(node.text or "" for node in cell.iter(f"{{{spreadsheet_ns}}}t"))
                else:
                    value_node = cell.find(f"{{{spreadsheet_ns}}}v")
                    raw_value = value_node.text if value_node is not None and value_node.text is not None else ""
                    if cell_type == "s" and raw_value:
                        value = shared_strings[int(raw_value)]
                    else:
                        value = raw_value
                values[index] = value
            if values:
                width = max(values) + 1
                rows.append([values.get(index, "") for index in range(width)])

    header_index = next((index for index, row in enumerate(rows) if row and row[0] == "Metric Key"), None)
    if header_index is None:
        return []
    headers = rows[header_index]
    records = []
    for row in rows[header_index + 1:]:
        if not row or not row[0]:
            continue
        padded_row = row + [""] * (len(headers) - len(row))
        records.append(dict(zip(headers, padded_row)))
    return records

class ProjectRAGSystem:
    AIRPORT_ALIASES = {
        "dallas fort worth": "DFW", "dallas": "DFW",
        "atlanta": "ATL",
        "chicago o hare": "ORD", "chicago o'hare": "ORD", "chicago": "ORD",
        "denver": "DEN",
        "los angeles": "LAX",
        "charlotte": "CLT",
        "new york jfk": "JFK", "kennedy": "JFK",
        "new york laguardia": "LGA", "laguardia": "LGA",
        "san francisco": "SFO",
        "phoenix": "PHX",
    }
    CARRIER_ALIASES = {
        "american airlines": "AA", "american": "AA",
        "delta air lines": "DL", "delta airlines": "DL", "delta": "DL",
        "united airlines": "UA", "united": "UA",
        "southwest airlines": "WN", "southwest": "WN",
        "jetblue airways": "B6", "jetblue": "B6",
        "alaska airlines": "AS", "alaska": "AS",
        "spirit airlines": "NK", "spirit": "NK",
        "frontier airlines": "F9", "frontier": "F9",
    }

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.chunks: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.audit_data = {}
        self.metrics_data = {}
        self.excel_knowledge: List[Dict[str, str]] = []
        self.is_indexed = False
        self.load_metadata()
        self.build_index()

    def load_metadata(self):
        """Loads factual project numbers for instant, direct accuracy."""
        audit_file = os.path.join(self.base_dir, "data", "processed", "cleaning_audit_summary.json")
        if os.path.exists(audit_file):
            try:
                with open(audit_file, "r", encoding="utf-8") as f:
                    self.audit_data = json.load(f)
            except Exception:
                pass

        metrics_file = os.path.join(self.base_dir, "models", "model_evaluation_metrics.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    self.metrics_data = json.load(f)
            except Exception:
                pass

        results_workbook = os.path.join(self.base_dir, "data", "processed", "ai_analyst_results.xlsx")
        if os.path.exists(results_workbook):
            try:
                self.excel_knowledge = load_xlsx_sheet_records(results_workbook, "Chat Results")
            except (OSError, KeyError, ValueError, zipfile.BadZipFile, ET.ParseError) as exc:
                print(f"Warning: could not read AI Analyst results workbook: {exc}")

    @staticmethod
    def _normalize_question(text: str) -> str:
        normalized = text.lower().replace("cleared", "cleaned")
        normalized = re.sub(r"\bmeann\b", "mean", normalized)
        words = re.findall(r"[a-z0-9]+", normalized)
        stop_words = {
            "a", "an", "and", "are", "can", "do", "does", "for", "from",
            "give", "how", "is", "me", "of", "please", "show", "the", "to", "what"
        }
        return " ".join(word for word in words if word not in stop_words)

    @staticmethod
    def _sources_requested(query: str) -> bool:
        """Returns True only when the viewer explicitly asks for supporting sources."""
        normalized = query.lower()
        explicit_terms = [
            "source", "sources", "citation", "citations", "evidence",
            "where did this come from", "where does this come from",
            "according to what", "provide references", "show references",
            "list references", "cite references", "references for",
        ]
        return any(phrase in normalized for phrase in explicit_terms)

    @staticmethod
    def _has_project_context(query: str) -> bool:
        """Checks that a general question identifies a project or aviation topic."""
        normalized = query.lower()
        project_terms = [
            "taxi", "flight", "aircraft", "airport", "airline", "carrier",
            "route", "sector", "origin", "destination", "departure", "delay",
            "distance", "congestion", "pushback", "queue", "runway", "model",
            "prediction", "forecast", "random forest", "xgboost", "regression",
            "mae", "rmse", "r2", "r-squared", "accuracy", "error", "feature",
            "dataset", "record", "clean", "outlier", "retention", "dashboard",
            "chart", "graph", "plot", "metric", "percentile", "month", "hour",
            "wheels-off", "a-cdm", "dispatch",
        ]
        return any(term in normalized for term in project_terms)

    def _answer_from_excel(self, query: str) -> Optional[Dict[str, Any]]:
        normalized_query = self._normalize_question(query)
        query_tokens = set(normalized_query.split())
        if not query_tokens:
            return None

        best_match = None
        best_score = 0.0
        for record in self.excel_knowledge:
            aliases = record.get("Question Aliases", "").split(";")
            for alias in aliases:
                normalized_alias = self._normalize_question(alias)
                alias_tokens = set(normalized_alias.split())
                if not alias_tokens:
                    continue
                if normalized_alias == normalized_query or normalized_alias in normalized_query:
                    score = 1.0
                else:
                    score = len(query_tokens & alias_tokens) / max(len(query_tokens), len(alias_tokens))
                if score > best_score:
                    best_score = score
                    best_match = record

        if best_match and best_score >= 0.6:
            source_sheet = best_match.get("Source Sheet", "Results")
            source_metric = best_match.get("Source Metric", best_match.get("Metric Key", "Metric"))
            return {
                "answer": best_match.get("Answer", ""),
                "sources": [f"[ai_analyst_results.xlsx - {source_sheet}: {source_metric}]"],
                "confidence": "High"
            }
        return None

    @staticmethod
    def _model_key_from_query(query: str) -> Optional[str]:
        """Maps common model names and abbreviations to metrics JSON keys."""
        q_lower = query.lower()
        if re.search(r"\brf\b", q_lower):
            return "Random_Forest_Regressor"
        if re.search(r"\bxgb\b", q_lower):
            return "XGBoost_Regressor"
        if re.search(r"\blr\b", q_lower):
            return "Linear_Regression_Baseline"
        aliases = {
            "Random_Forest_Regressor": [
                "random forest", "random forests", "rf model", "rf regressor"
            ],
            "XGBoost_Regressor": [
                "xgboost", "xgb model", "xgb regressor", "boosted model",
                "gradient boosting model"
            ],
            "Linear_Regression_Baseline": [
                "linear regression", "linear model", "lr model", "baseline model"
            ],
        }
        for model_key, model_aliases in aliases.items():
            if any(alias in q_lower for alias in model_aliases):
                return model_key
        return None

    @staticmethod
    def _display_model_name(model_key: str) -> str:
        return {
            "Random_Forest_Regressor": "Random Forest",
            "XGBoost_Regressor": "XGBoost",
            "Linear_Regression_Baseline": "Linear Regression",
        }.get(model_key, model_key.replace("_", " "))

    def _answer_from_model_knowledge(self, query: str) -> Optional[Dict[str, Any]]:
        """Answers broad model, metric, feature, and risk-matrix questions from authoritative data."""
        q_lower = self._normalize_calculation_query(query)
        models = self.metrics_data.get("models", {})
        model_key = self._model_key_from_query(q_lower)

        sensitivity_terms = [
            "congestion sensitivity", "sensitivity curve", "live congestion curve",
            "surface traffic sensitivity", "congestion response curve"
        ]
        if any(term in q_lower for term in sensitivity_terms):
            return {
                "answer": (
                    "**Live Congestion Sensitivity Curve**\n\n"
                    "This is a **what-if analysis** showing how the selected model's taxi-out "
                    "prediction changes as origin pushback congestion changes. The dashboard keeps "
                    "the selected flight's other inputs fixed, varies one-hour pushbacks from "
                    "**1 to 35**, sets 30-minute congestion to approximately half of each hourly "
                    "value, and recalculates the prediction.\n\n"
                    "- **X-axis:** origin pushbacks during the previous hour.\n"
                    "- **Y-axis:** predicted taxi-out time in minutes.\n"
                    "- **Dashed vertical line:** the currently selected congestion value.\n\n"
                    "A rising or steeper section means the model is more sensitive to added surface "
                    "traffic in that range. A flatter section means the prediction changes little. "
                    "The curve describes model behavior for the selected scenario; it is not an "
                    "observed time series, a causal estimate, or a forecast of future traffic volume."
                ),
                "sources": [
                    "[streamlit_app.py - get_cached_sensitivity_curve and Live Congestion Sensitivity Curve]",
                    "[docs/ai_analyst_knowledge.md - Live congestion sensitivity curve]",
                ],
                "confidence": "High",
            }

        confusion_terms = [
            "confusion matrix", "confusion table", "risk matrix",
            "classification matrix", "risk-band matrix", "risk band matrix",
            "risk heatmap", "risk-band heatmap", "risk band heatmap"
        ]
        if any(term in q_lower for term in confusion_terms):
            return {
                "answer": (
                    "**Operational Risk Confusion Matrix**\n\n"
                    "It compares the **actual taxi-out risk band** with the "
                    "**predicted taxi-out risk band** for test flights. Rows are actual bands "
                    "and columns are predicted bands. Diagonal cells are correct band assignments; "
                    "off-diagonal cells show assignment to another operational band.\n\n"
                    "The bands are **Normal (<20 min)**, **Moderate (20 to <30 min)**, "
                    "**High (30 to <45 min)**, and **Severe (45+ min)**. This is a "
                    "classification view of continuous regression predictions, not a separately trained classifier."
                ),
                "sources": [
                    "[streamlit_app.py - Operational Risk Confusion Matrix]",
                    "[docs/ai_analyst_knowledge.md - Confusion matrix]",
                ],
                "confidence": "High",
            }

        feature_terms = [
            "feature importance", "feature importances", "important feature",
            "important features", "top feature", "top features", "key feature",
            "key features", "top predictor", "top predictors", "feature ranking",
            "important predictor", "important predictors", "important variable",
            "important variables", "influential feature", "influential features",
            "most influential", "what affects taxi", "what drives taxi"
        ]
        asks_features = any(term in q_lower for term in feature_terms)
        if asks_features and models:
            selected_keys = [model_key] if model_key else [
                key for key in ["Random_Forest_Regressor", "XGBoost_Regressor"] if key in models
            ]
            limit = self._requested_limit(query, default=5)
            sections = []
            for selected_key in selected_keys:
                details = models.get(selected_key, {})
                model_name = self._display_model_name(selected_key)
                if selected_key == "Linear_Regression_Baseline":
                    coefficients = details.get("top_coefficients", [])[:limit]
                    if coefficients:
                        lines = [f"**{model_name} largest coefficient magnitudes:**"]
                        lines.extend(
                            f"- **{item['feature']}**: {item['coefficient']:+.4f}"
                            for item in coefficients
                        )
                        lines.append("Coefficients show fitted direction and magnitude after preprocessing; they are not tree feature importances.")
                        sections.append("\n".join(lines))
                    continue

                importances = details.get("top_importances", [])[:limit]
                if not importances:
                    continue
                lines = [f"**{model_name} top {len(importances)} features:**"]
                lines.extend(
                    f"- **{item['feature']}**: {item['importance'] * 100:.2f}%"
                    for item in importances
                )
                sections.append("\n".join(lines))

            if sections:
                return {
                    "answer": "\n\n".join(sections) + (
                        "\n\nFeature importance describes how the fitted model used inputs; it does not prove causation."
                    ),
                    "sources": ["[models/model_evaluation_metrics.json - trained model feature importance]"],
                    "confidence": "High",
                }

        metric_definitions = []
        metric_map = [
            (["mae", "mean absolute error"], "**MAE** is the average absolute prediction error in minutes. Lower is better."),
            (["rmse", "root mean squared error"], "**RMSE** gives larger errors extra weight before returning the result to minutes. Lower is better."),
            (["r2", "r-squared", "r squared", "coefficient of determination"], "**R-squared** compares explained variance with a mean-prediction baseline. Higher is better; values can remain low when important operational signals are unavailable."),
            (["within 3", "within three", "3 minute accuracy"], "**Within 3 minutes** is the share of predictions no more than 3 minutes above or below the observed taxi-out time. Higher is better."),
            (["within 5", "within five", "5 minute accuracy"], "**Within 5 minutes** is the share of predictions no more than 5 minutes above or below the observed taxi-out time. Higher is better."),
            (["operational accuracy", "prediction accuracy"], "**Operational accuracy** is reported as the percentage of predictions within a stated tolerance, such as plus or minus 3 or 5 minutes."),
        ]
        asks_metric_definition = any(
            term in q_lower for term in ["what is", "what does", "define", "explain", "meaning", "evaluation metric"]
        )
        if asks_metric_definition:
            for aliases, definition in metric_map:
                if any(alias in q_lower for alias in aliases):
                    metric_definitions.append(definition)
            if "evaluation metric" in q_lower and not metric_definitions:
                metric_definitions = [definition for _, definition in metric_map]
            if metric_definitions:
                return {
                    "answer": "\n\n".join(metric_definitions),
                    "sources": ["[docs/ai_analyst_knowledge.md - Model evaluation metrics]"],
                    "confidence": "High",
                }

        hyperparameter_terms = [
            "hyperparameter", "hyperparameters", "model parameter", "model parameters",
            "model setting", "model settings", "how is the model configured"
        ]
        if model_key and any(term in q_lower for term in hyperparameter_terms):
            parameters = models.get(model_key, {}).get("hyperparameters", {})
            if parameters:
                lines = [f"**{self._display_model_name(model_key)} hyperparameters:**"]
                lines.extend(f"- **{name}**: {value}" for name, value in parameters.items())
                return {
                    "answer": "\n".join(lines),
                    "sources": ["[models/model_evaluation_metrics.json - hyperparameters]"],
                    "confidence": "High",
                }

        comparison_terms = [
            "best model", "better model", "performing better", "performs better",
            "top model", "strongest model", "most accurate model", "which model",
            "best algorithm", "better algorithm", "which algorithm", "which regressor",
            "compare model", "compare the model", "compare algorithm", "lowest error",
            "lowest mae", "lowest rmse", "highest r2", "highest r-squared",
            "highest accuracy", "model comparison", "model performance"
        ]
        metric_terms = [
            "mae", "rmse", "r2", "r-squared", "r squared", "accuracy",
            "performance", "perform", "score", "error"
        ]
        asks_comparison = any(term in q_lower for term in comparison_terms)
        asks_model_metrics = model_key is not None and any(term in q_lower for term in metric_terms)
        if models and (asks_comparison or asks_model_metrics):
            selected_models = models.items() if asks_comparison else [(model_key, models.get(model_key, {}))]
            results = []
            for selected_key, details in selected_models:
                test = details.get("test", {})
                if "MAE_minutes" not in test:
                    continue
                results.append((selected_key, test))
            if results:
                results.sort(key=lambda item: item[1]["MAE_minutes"])
                lines = ["**Out-of-time test performance:**", ""]
                for selected_key, test in results:
                    lines.append(
                        f"- **{self._display_model_name(selected_key)}:** "
                        f"MAE **{test['MAE_minutes']:.3f} min**, "
                        f"RMSE **{test['RMSE_minutes']:.3f} min**, "
                        f"R-squared **{test['R2_Score']:.4f}**, "
                        f"within 5 minutes **{test['Pct_Within_5_Min']:.2f}%**."
                    )
                if asks_comparison:
                    lines.extend([
                        "",
                        f"**Best current test model:** {self._display_model_name(results[0][0])}, based on the lowest MAE."
                    ])
                return {
                    "answer": "\n".join(lines),
                    "sources": ["[models/model_evaluation_metrics.json - out-of-time test metrics]"],
                    "confidence": "High",
                }

        return None

    def _find_airports(self, query: str, flight_data: Any) -> List[str]:
        """Returns airport codes in the order they appear in the question."""
        query_lower = query.lower()
        matches = []
        available_codes = set()
        if flight_data is not None:
            try:
                available_codes.update(flight_data["ORIGIN"].dropna().astype(str).str.upper().unique())
                available_codes.update(flight_data["DEST"].dropna().astype(str).str.upper().unique())
            except (KeyError, AttributeError):
                pass

        for alias, code in self.AIRPORT_ALIASES.items():
            for match in re.finditer(rf"\b{re.escape(alias)}\b", query_lower):
                matches.append((match.start(), code))
        for code in available_codes:
            for match in re.finditer(rf"\b{re.escape(code.lower())}\b", query_lower):
                matches.append((match.start(), code))

        ordered_codes = []
        for _, code in sorted(matches):
            if code not in ordered_codes:
                ordered_codes.append(code)
        return ordered_codes

    def _find_carrier(self, query: str, flight_data: Any) -> Optional[str]:
        """Returns a carrier code explicitly mentioned in the question."""
        query_lower = query.lower()
        matches = []
        available_codes = set()
        if flight_data is not None:
            try:
                available_codes.update(
                    flight_data["OP_UNIQUE_CARRIER"].dropna().astype(str).str.upper().unique()
                )
            except (KeyError, AttributeError):
                pass

        for alias, code in self.CARRIER_ALIASES.items():
            match = re.search(rf"\b{re.escape(alias)}\b", query_lower)
            if match:
                matches.append((match.start(), code))
        for code in available_codes:
            match = re.search(rf"\b{re.escape(code.lower())}\b", query_lower)
            if match:
                matches.append((match.start(), code))

        return min(matches)[1] if matches else None

    @staticmethod
    def _requested_limit(query: str, default: int = 5) -> int:
        """Returns a safe result count for ranking questions such as 'top 3 airports'."""
        match = re.search(r"\b(?:top|bottom|first|last)\s+(\d+)\b", query.lower())
        return max(1, min(int(match.group(1)), 10)) if match else default

    def _answer_from_route_data(self, query: str, flight_data: Any) -> Optional[Dict[str, Any]]:
        """Answers route count and route-level taxi-out ranking questions."""
        q_lower = self._normalize_calculation_query(query)
        if not any(term in q_lower for term in ["route", "routes", "city pair", "origin destination"]):
            return None

        asks_share = any(term in q_lower for term in ["percent", "%", "share", "proportion", "portion"])
        asks_distance = any(term in q_lower for term in ["distance", "miles", "mileage", "longest", "shortest"])
        asks_taxi = any(term in q_lower for term in ["taxi", "ground time", "take off", "takeoff"])
        asks_count = any(term in q_lower for term in [
            "busiest", "most flights", "most frequent", "most common",
            "highest number", "flight count", "top ", "bottom ", "rank", "ranking",
        ])
        if asks_share or (not asks_distance and not asks_taxi and not asks_count):
            return None

        try:
            columns = ["ORIGIN", "DEST"] + (["TAXI_OUT"] if asks_taxi else [])
            values = flight_data[columns].copy()
            if asks_taxi:
                values["TAXI_OUT"] = values["TAXI_OUT"].astype(float)
                values = values.dropna(subset=["ORIGIN", "DEST", "TAXI_OUT"])
            else:
                values = values.dropna(subset=["ORIGIN", "DEST"])
        except (KeyError, AttributeError, TypeError, ValueError):
            return None
        if values.empty:
            return None

        if asks_count and not asks_distance and not asks_taxi:
            summary = values.groupby(["ORIGIN", "DEST"]).size().sort_values(ascending=False)
            limit = self._requested_limit(
                query,
                default=5 if any(term in q_lower for term in ["top ", "rank", "ranking", "routes"]) else 1,
            )
            lines = ["**Busiest Routes by cleaned flight count:**", ""]
            for rank, ((origin, destination), count) in enumerate(summary.head(limit).items(), start=1):
                lines.append(f"{rank}. **{origin} -> {destination}** - **{int(count):,} cleaned flights**")
            lines.append("\nRoute direction is counted separately (for example, DFW -> ATL differs from ATL -> DFW).")
            return {
                "answer": "\n".join(lines),
                "sources": ["[data/processed/cleaned_flights.csv - route flight-count aggregation]"],
                "confidence": "High",
            }

        if not asks_taxi:
            return None
        summary = values.groupby(["ORIGIN", "DEST"])["TAXI_OUT"].agg(
            Mean="mean", Median="median", Flights="count"
        )
        if summary.empty:
            return None
        is_lowest = any(term in q_lower for term in ["lowest", "smallest", "minimum", "shortest", "bottom "])
        is_ranking = any(term in q_lower for term in ["highest", "lowest", "top ", "bottom ", "rank", "ranking"])
        if not is_ranking:
            return None
        asks_many = any(term in q_lower for term in ["top ", "bottom ", "rank", "ranking"])
        ranked = summary.sort_values("Mean", ascending=is_lowest).head(
            self._requested_limit(query, default=5 if asks_many else 1)
        )
        direction = "Lowest" if is_lowest else "Highest"
        lines = [f"**{direction} Routes by average taxi-out:**", ""]
        for rank, ((origin, destination), row) in enumerate(ranked.iterrows(), start=1):
            lines.append(
                f"{rank}. **{origin} -> {destination}** - **{row['Mean']:.1f} minutes** "
                f"({int(row['Flights']):,} cleaned flights)"
            )
        return {
            "answer": "\n".join(lines),
            "sources": ["[data/processed/cleaned_flights.csv - route taxi-out aggregation]"],
            "confidence": "High",
        }

    def _answer_from_distance_data(self, query: str, flight_data: Any) -> Optional[Dict[str, Any]]:
        """Answers direct stage-distance and route-distance questions from live records."""
        q_lower = self._normalize_calculation_query(query)
        if not any(term in q_lower for term in ["distance", "miles", "mileage", "longest route", "shortest route"]):
            return None

        try:
            values = flight_data[["ORIGIN", "DEST", "DISTANCE"]].copy()
            values["DISTANCE"] = values["DISTANCE"].astype(float)
            values = values.dropna(subset=["DISTANCE"])
        except (KeyError, AttributeError, TypeError, ValueError):
            return None
        if values.empty:
            return None

        asks_route = any(term in q_lower for term in ["route", "routes", "city pair"])
        asks_lowest = any(term in q_lower for term in ["shortest", "lowest", "smallest", "minimum", "min"])
        asks_extreme = any(term in q_lower for term in [
            "longest", "shortest", "highest", "lowest", "maximum", "minimum", "max", "min"
        ])
        if asks_route:
            summary = values.groupby(["ORIGIN", "DEST"])["DISTANCE"].agg(
                Mean="mean", Median="median", Flights="count"
            )
            ranked = summary.sort_values("Mean", ascending=asks_lowest)
            limit = self._requested_limit(query, default=1 if asks_extreme else 5)
            direction = "Shortest" if asks_lowest else "Longest"
            lines = [f"**{direction} Routes by average stage distance:**", ""]
            for rank, ((origin, destination), row) in enumerate(ranked.head(limit).iterrows(), start=1):
                lines.append(
                    f"{rank}. **{origin} -> {destination}** - **{row['Mean']:.0f} miles** "
                    f"({int(row['Flights']):,} cleaned flights)"
                )
            return {
                "answer": "\n".join(lines),
                "sources": ["[data/processed/cleaned_flights.csv - route distance aggregation]"],
                "confidence": "High",
            }

        statistic = values["DISTANCE"].median() if "median" in q_lower else values["DISTANCE"].mean()
        statistic_label = "Median" if "median" in q_lower else "Average"
        if asks_extreme:
            statistic = values["DISTANCE"].min() if asks_lowest else values["DISTANCE"].max()
            statistic_label = "Minimum" if asks_lowest else "Maximum"
        return {
            "answer": (
                f"**{statistic_label} Stage Distance:** **{statistic:.0f} miles**\n\n"
                f"Based on **{len(values):,} cleaned flight records**."
            ),
            "sources": ["[data/processed/cleaned_flights.csv - stage-distance aggregation]"],
            "confidence": "High",
        }
    @staticmethod
    def _normalize_calculation_query(text: str) -> str:
        """Normalizes a small set of common calculation-query spelling variants."""
        normalized = text.lower().strip()
        for misspelling, correction in {
            "hightest": "highest",
            "higest": "highest",
            "averge": "average",
            "avarge": "average",
            "conjusted": "congested",
            "meann": "mean",
            "perfoming": "performing",
            "performace": "performance",
            "confussion": "confusion",
            "importace": "importance",
            "percents": "percent",
            "percentage": "percent",
            "quantile": "percentile",
        }.items():
            normalized = re.sub(rf"\b{misspelling}\b", correction, normalized)
        return normalized

    def _answer_from_record_share(
        self,
        query: str,
        flight_data: Any,
    ) -> Optional[Dict[str, Any]]:
        """Answers record-share questions from cleaned flight records."""
        q_lower = self._normalize_calculation_query(query)
        share_terms = ["percent", "%", "share", "proportion", "portion"]
        record_terms = ["record", "records", "flight", "flights"]
        competing_terms = [
            "cancelled", "canceled", "diverted", "removed", "retention",
            "departure delay", "distance", "model error",
        ]
        sector_terms = ["sector", "sectors", "route", "routes", "city pair"]
        asks_sector = any(term in q_lower for term in sector_terms)
        asks_sector_ranking = asks_sector and any(
            term in q_lower for term in ["busiest", "most", "highest", "top", "rank", "ranking"]
        )
        if (
            (not any(term in q_lower for term in share_terms) and not asks_sector_ranking)
            or (not any(term in q_lower for term in record_terms) and not asks_sector_ranking)
            or any(term in q_lower for term in competing_terms)
        ):
            return None

        if asks_sector:
            try:
                denominator = len(flight_data)
                sector_values = flight_data[["ORIGIN", "DEST"]].dropna()
                sector_counts = sector_values.groupby(["ORIGIN", "DEST"]).size().sort_values(ascending=False)
            except (KeyError, AttributeError, TypeError, ValueError):
                return None
            if denominator == 0 or sector_counts.empty:
                return None
            (origin, destination), count = sector_counts.index[0], int(sector_counts.iloc[0])
            percentage = count / denominator * 100.0
            return {
                "answer": (
                    f"**Busiest Sector:** **{origin} -> {destination}**\n\n"
                    f"It contributes **{percentage:.2f}%** of cleaned flight records "
                    f"(**{count:,} of {denominator:,} records**).\n\n"
                    "Sector direction is counted separately; the reverse sector is a different route."
                ),
                "sources": ["[data/processed/cleaned_flights.csv - busiest sector record share]"],
                "confidence": "High",
            }

        airports = self._find_airports(query, flight_data)
        carrier_code = self._find_carrier(query, flight_data)
        if not airports and not carrier_code:
            return None

        use_destination = any(
            term in q_lower for term in ["destination", "arrival", "arrivals"]
        )
        group_column = "DEST" if use_destination else "ORIGIN"
        group_label = "destination" if use_destination else "origin airport"

        try:
            denominator = len(flight_data)
            if denominator == 0:
                return None
            if airports:
                selected_values = airports
                group_values = flight_data[group_column].astype(str).str.upper()
            else:
                selected_values = [carrier_code]
                group_values = flight_data["OP_UNIQUE_CARRIER"].astype(str).str.upper()
            carrier_values = None
            if carrier_code and airports:
                carrier_values = flight_data["OP_UNIQUE_CARRIER"].astype(str).str.upper()
        except (KeyError, AttributeError, TypeError):
            return None

        lines = ["**Share of Cleaned Flight Records:**", ""]
        for selected_value in selected_values:
            mask = group_values == selected_value
            if carrier_values is not None:
                mask &= carrier_values == carrier_code
            count = int(mask.sum())
            percentage = count / denominator * 100.0
            qualifier = f" at {group_label}" if airports else ""
            if carrier_values is not None:
                qualifier += f" for carrier {carrier_code}"
            lines.append(
                f"- **{selected_value}**: **{percentage:.2f}%** "
                f"({count:,} of {denominator:,} cleaned records{qualifier})"
            )

        return {
            "answer": "\n".join(lines),
            "sources": [
                f"[data/processed/cleaned_flights.csv - cleaned record share by {group_label}]"
            ],
            "confidence": "High",
        }
    def _answer_from_record_count_ranking(
        self,
        query: str,
        flight_data: Any,
    ) -> Optional[Dict[str, Any]]:
        """Answers airport or carrier record-count ranking questions."""
        q_lower = self._normalize_calculation_query(query)
        record_terms = ["record", "records", "flight", "flights"]
        ranking_terms = [
            "highest", "largest", "maximum", "max", "most", "top ",
            "lowest", "smallest", "minimum", "min", "least", "bottom ",
            "rank", "ranking",
        ]
        competing_metric_terms = [
            "taxi", "average", "mean", "median", "percentile", "minute",
            "time", "duration", "delay", "distance", "model error",
        ]
        if (
            not any(term in q_lower for term in record_terms)
            or not any(term in q_lower for term in ranking_terms)
            or any(term in q_lower for term in competing_metric_terms)
        ):
            return None

        mentions_airport = any(
            term in q_lower for term in ["airport", "airports", "origin", "destination", "arrival"]
        )
        mentions_carrier = any(
            term in q_lower for term in ["airline", "airlines", "carrier"]
        ) or self._find_carrier(query, flight_data) is not None
        if not mentions_airport and not mentions_carrier:
            return None

        use_destination = any(
            term in q_lower for term in ["destination", "arrival", "arrivals"]
        )
        if mentions_airport:
            group_column = "DEST" if use_destination else "ORIGIN"
            group_label = "Airport" if not use_destination else "Destination Airport"
        else:
            group_column = "OP_UNIQUE_CARRIER"
            group_label = "Carrier"

        try:
            values = flight_data[group_column].dropna().astype(str).str.upper()
            counts = values.value_counts()
        except (KeyError, AttributeError, TypeError, ValueError):
            return None
        if counts.empty:
            return None

        is_lowest = any(term in q_lower for term in ["lowest", "smallest", "minimum", "min", "least", "bottom "])
        asks_many = any(term in q_lower for term in ["top ", "bottom ", "rank", "ranking"])
        limit = self._requested_limit(query, default=5 if asks_many else 1)
        ranked = counts.sort_values(ascending=is_lowest).head(limit)
        direction = "Lowest" if is_lowest else "Highest"
        lines = [f"**{direction} {group_label} by cleaned record count:**", ""]
        for rank, (group_value, count) in enumerate(ranked.items(), start=1):
            lines.append(f"{rank}. **{group_value}** — **{int(count):,} cleaned records**")
        return {
            "answer": "\n".join(lines),
            "sources": [f"[data/processed/cleaned_flights.csv - {group_label.lower()} record-count aggregation]"],
            "confidence": "High",
        }
    def _answer_from_generic_aggregation(self, query: str, flight_data: Any) -> Optional[Dict[str, Any]]:
        """Builds common count/mean/ranking answers from query dimensions dynamically."""
        if flight_data is None:
            return None
        q_lower = self._normalize_calculation_query(query)

        ranking_terms = [
            "busiest", "most", "highest", "largest", "maximum", "max", "top ",
            "rank", "ranking", "lowest", "smallest", "minimum", "min", "least",
        ]
        summary_terms = ["average", "avg", "mean", "median", "summary", "statistics"]
        asks_ranking = any(term in q_lower for term in ranking_terms)
        asks_summary = any(term in q_lower for term in summary_terms)

        asks_month = (
            any(term in q_lower for term in ["monthly", "per month", "by month", "each month", "any month"])
            or bool(re.search(r"\bmonths?\b", q_lower))
        )
        asks_day = any(term in q_lower for term in ["day", "date", "daily"])
        asks_route = any(term in q_lower for term in ["route", "routes", "sector", "sectors", "city pair"])
        asks_carrier = any(term in q_lower for term in ["carrier", "airline", "airlines"])
        asks_airport = any(term in q_lower for term in ["airport", "airports", "origin", "destination"])
        asks_hour = any(term in q_lower for term in ["hour", "hourly", "time of day"])

        asks_count = any(term in q_lower for term in [
            "busiest", "most flights", "most frequent", "flight count", "number of flights",
            "records", "flights",
        ]) or (asks_ranking and not any(term in q_lower for term in ["taxi", "distance", "miles", "delay", "time"]))
        asks_taxi = any(term in q_lower for term in ["taxi", "ground time", "take off", "takeoff"])
        asks_distance = any(term in q_lower for term in ["distance", "miles", "mileage"])
        if not (asks_ranking or asks_summary) or not (asks_month or asks_day or asks_route or asks_carrier or asks_airport or asks_hour):
            return None
        if not (asks_count or asks_taxi or asks_distance):
            return None

        metric_column = None
        metric_label = None
        statistic = "count"
        if asks_taxi:
            metric_column = "TAXI_OUT"
            metric_label = "Mean taxi-out"
            statistic = "median" if "median" in q_lower else "mean"
        elif asks_distance:
            metric_column = "DISTANCE"
            metric_label = "Mean distance"
            statistic = "median" if "median" in q_lower else "mean"
        else:
            metric_label = "Flights"

        try:
            required_columns = ["ORIGIN", "DEST"]
            if asks_month or asks_day:
                required_columns.append("FL_DATE")
            if asks_carrier:
                required_columns.append("OP_UNIQUE_CARRIER")
            if asks_hour:
                required_columns.append("DEP_HOUR" if "DEP_HOUR" in flight_data.columns else "CRS_DEP_TIME")
            if metric_column:
                required_columns.append(metric_column)
            values = flight_data[list(dict.fromkeys(required_columns))].copy()
            if "FL_DATE" in values.columns:
                values["FL_DATE"] = pd.to_datetime(values["FL_DATE"], errors="coerce")
                values = values.dropna(subset=["FL_DATE"])
                values["YEAR"] = values["FL_DATE"].dt.year
                values["MONTH_NUM"] = values["FL_DATE"].dt.month
                values["MONTH"] = values["FL_DATE"].dt.strftime("%b")
                values["DATE"] = values["FL_DATE"].dt.strftime("%Y-%m-%d")
            if asks_hour and "DEP_HOUR" not in values.columns:
                values["DEP_HOUR"] = (pd.to_numeric(values["CRS_DEP_TIME"], errors="coerce") // 100).clip(0, 23)
            values = values.dropna(subset=["ORIGIN", "DEST"])
            if metric_column:
                values[metric_column] = pd.to_numeric(values[metric_column], errors="coerce")
                values = values.dropna(subset=[metric_column])
        except (KeyError, AttributeError, TypeError, ValueError):
            return None
        if values.empty:
            return None

        group_columns = []
        period_columns = []
        if asks_month:
            group_columns.extend(["YEAR", "MONTH_NUM", "MONTH"])
            period_columns.extend(["YEAR", "MONTH_NUM"])
        if asks_day:
            group_columns.append("DATE")
        if asks_route:
            group_columns.extend(["ORIGIN", "DEST"])
        elif asks_airport:
            group_columns.append("DEST" if "destination" in q_lower or "arrival" in q_lower else "ORIGIN")
        if asks_carrier:
            group_columns.append("OP_UNIQUE_CARRIER")
        if asks_hour:
            group_columns.append("DEP_HOUR")
        group_columns = list(dict.fromkeys(group_columns))
        if not group_columns:
            return None

        try:
            if statistic == "count":
                summary = values.groupby(group_columns).size().reset_index(name="Value")
            else:
                summary = values.groupby(group_columns)[metric_column].agg(statistic).reset_index(name="Value")
                counts = values.groupby(group_columns).size().reset_index(name="Flights")
                summary = summary.merge(counts, on=group_columns, how="left")
        except (KeyError, AttributeError, TypeError, ValueError):
            return None
        if summary.empty:
            return None

        is_lowest = any(term in q_lower for term in ["lowest", "smallest", "minimum", "min", "least"])
        sort_ascending = is_lowest
        period_rank = bool(
            asks_month
            and asks_ranking
            and any(term in q_lower for term in ["by month", "each month", "per month", "any month"])
        )
        asks_many = any(term in q_lower for term in ["top ", "rank", "ranking", "each ", "every "])
        limit = self._requested_limit(query, default=5 if asks_many else 1)

        if statistic == "count":
            denominator_columns = period_columns if period_columns else []
            if denominator_columns:
                denominators = values.groupby(denominator_columns).size().reset_index(name="Denominator")
                summary = summary.merge(denominators, on=denominator_columns, how="left")
                summary["Share_Pct"] = summary["Value"] / summary["Denominator"] * 100.0
            else:
                summary["Share_Pct"] = summary["Value"] / len(values) * 100.0

        sort_columns = period_columns + ["Value"] if period_rank else ["Value"]
        ascending = [True] * len(period_columns) + [sort_ascending] if period_rank else [sort_ascending]
        summary = summary.sort_values(sort_columns, ascending=ascending)
        if period_rank:
            summary = summary.groupby(period_columns, as_index=False).head(limit)
        else:
            summary = summary.head(limit)

        headers = []
        if asks_month:
            headers.append("Month")
        if asks_day:
            headers.append("Date")
        if asks_route:
            headers.append("Sector")
        elif asks_airport:
            headers.append("Airport")
        if asks_carrier:
            headers.append("Carrier")
        if asks_hour:
            headers.append("Departure hour")
        headers.append(metric_label)
        if statistic == "count":
            headers.append("Share")

        lines = ["**Analytical Result:**", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in summary.iterrows():
            cells = []
            if asks_month:
                cells.append(f"{int(row['YEAR'])} {row['MONTH']}" if summary["YEAR"].nunique() > 1 else str(row["MONTH"]))
            if asks_day:
                cells.append(str(row["DATE"]))
            if asks_route:
                cells.append(f"{row['ORIGIN']} -> {row['DEST']}")
            elif asks_airport:
                airport_column = "DEST" if "destination" in q_lower or "arrival" in q_lower else "ORIGIN"
                cells.append(str(row[airport_column]))
            if asks_carrier:
                cells.append(str(row["OP_UNIQUE_CARRIER"]))
            if asks_hour:
                cells.append(str(int(row["DEP_HOUR"])))
            cells.append(f"{row['Value']:.1f}" if statistic != "count" else f"{int(row['Value']):,}")
            if statistic == "count":
                cells.append(f"{row['Share_Pct']:.2f}%")
            lines.append("| " + " | ".join(cells) + " |")

        return {
            "answer": "\n".join(lines),
            "sources": ["[data/processed/cleaned_flights.csv - dynamic grouped aggregation]"],
            "confidence": "High",
        }
    def _answer_from_monthly_route_data(self, query: str, flight_data: Any) -> Optional[Dict[str, Any]]:
        """Answers monthly busiest-sector questions using route flight counts."""
        q_lower = self._normalize_calculation_query(query)
        asks_monthly = (
            any(term in q_lower for term in ["monthly", "per month", "by month", "each month"])
            or bool(re.search(r"\bmonths?\b", q_lower))
        )
        asks_sector = any(term in q_lower for term in ["sector", "sectors", "route", "routes", "city pair"])
        asks_busiest = any(term in q_lower for term in [
            "busiest", "most flights", "most frequent", "highest number", "flight count", "top", "rank"
        ])
        if not asks_monthly or not asks_sector or not asks_busiest:
            return None

        try:
            values = flight_data[["FL_DATE", "ORIGIN", "DEST"]].copy()
            values["FL_DATE"] = pd.to_datetime(values["FL_DATE"], errors="coerce")
            values = values.dropna(subset=["FL_DATE", "ORIGIN", "DEST"])
            values["YEAR"] = values["FL_DATE"].dt.year
            values["MONTH_NUM"] = values["FL_DATE"].dt.month
            values["MONTH"] = values["FL_DATE"].dt.strftime("%b")
            counts = (
                values.groupby(["YEAR", "MONTH_NUM", "MONTH", "ORIGIN", "DEST"])
                .size()
                .reset_index(name="Flights")
            )
            counts["Month_Total"] = counts.groupby(["YEAR", "MONTH_NUM"])["Flights"].transform("sum")
            counts["Share_Pct"] = counts["Flights"] / counts["Month_Total"] * 100.0
        except (KeyError, AttributeError, TypeError, ValueError):
            return None

        if counts.empty:
            return None
        ranked = counts.sort_values(
            ["YEAR", "MONTH_NUM", "Flights", "ORIGIN", "DEST"],
            ascending=[True, True, False, True, True],
        )
        leaders = ranked.groupby(["YEAR", "MONTH_NUM"], as_index=False).head(1)
        multiple_years = leaders["YEAR"].nunique() > 1
        lines = ["**Busiest Sector by Month:**", ""]
        lines.append("| Month | Busiest sector | Flights | Share of monthly records |")
        lines.append("| --- | --- | ---: | ---: |")
        for _, row in leaders.sort_values(["YEAR", "MONTH_NUM"]).iterrows():
            period = f"{int(row['YEAR'])} {row['MONTH']}" if multiple_years else row["MONTH"]
            lines.append(
                f"| {period} | **{row['ORIGIN']} -> {row['DEST']}** | "
                f"{int(row['Flights']):,} | {row['Share_Pct']:.2f}% |"
            )
        return {
            "answer": "\n".join(lines),
            "sources": ["[data/processed/cleaned_flights.csv - monthly sector flight-count aggregation]"],
            "confidence": "High",
        }
    def _answer_from_monthly_data(
        self,
        query: str,
        flight_data: Any,
    ) -> Optional[Dict[str, Any]]:
        """Answers month-level taxi-out statistic questions from the shared table."""
        q_lower = self._normalize_calculation_query(query)
        asks_monthly = (
            any(term in q_lower for term in ["monthly", "per month", "by month", "each month"])
            or bool(re.search(r"\bmonths?\b", q_lower))
        )
        competing_terms = [
            "departure delay", "flight duration", "flight time", "distance",
            "congestion", "model error", "sector", "sectors", "route", "routes",
            "city pair", "busiest", "flight count", "records",
        ]
        if not asks_monthly or any(term in q_lower for term in competing_terms):
            return None

        statistic_specs = [
            ("mean", "Mean", "Mean_Taxi_Out", ["mean", "average", "avg"]),
            ("median", "Median / P50", "P50_Taxi_Out", ["median", "p50", "50th percentile", "50th"]),
            ("p25", "P25", "P25_Taxi_Out", ["p25", "25th percentile", "25th"]),
            ("p75", "P75", "P75_Taxi_Out", ["p75", "75th percentile", "75th"]),
            ("p90", "P90", "P90_Taxi_Out", ["p90", "90th percentile", "90th"]),
            ("p95", "P95", "P95_Taxi_Out", ["p95", "95th percentile", "95th"]),
            ("max", "Maximum", "Max_Taxi_Out", ["max", "maximum", "highest", "largest", "peak"]),
            ("min", "Minimum", "Min_Taxi_Out", ["min", "minimum", "lowest", "smallest"]),
        ]
        percentile_numbers = set()
        percentile_patterns = [
            r"\bp[\s_-]?(\d{1,3})(?:st|nd|rd|th)?s?\b",
            r"\bpercentiles?\s+(?:of\s+)?(\d{1,3})s?\b",
            r"\b(\d{1,3})(?:st|nd|rd|th)?\s+percentiles?\b",
        ]
        for pattern in percentile_patterns:
            for match in re.finditer(pattern, q_lower):
                percentile = int(match.group(1))
                if 0 <= percentile <= 100:
                    percentile_numbers.add(percentile)

        dynamic_specs = [
            (f"p{percentile}", f"P{percentile}", f"P{percentile}_Taxi_Out", [])
            for percentile in sorted(percentile_numbers)
            if f"P{percentile}_Taxi_Out" not in {spec[2] for spec in statistic_specs}
        ]
        requested = [
            spec for spec in statistic_specs
            if any(alias in q_lower for alias in spec[3])
        ]
        requested_columns = {spec[2] for spec in requested}
        requested.extend(spec for spec in dynamic_specs if spec[2] not in requested_columns)
        asks_summary = any(term in q_lower for term in ["summary", "statistics", "descriptive", "all percentile"])
        if asks_summary:
            requested = statistic_specs + [spec for spec in dynamic_specs if spec[2] not in {item[2] for item in statistic_specs}]
        elif not requested:
            requested = statistic_specs
        monthly_data = flight_data
        airport_scope = ""
        airports = self._find_airports(query, flight_data)
        if airports:
            airport_code = airports[0]
            airport_column = "DEST" if any(
                term in q_lower for term in ["destination", "arrival"]
            ) else "ORIGIN"
            try:
                airport_values = flight_data[airport_column].astype(str).str.upper()
                monthly_data = flight_data.loc[airport_values == airport_code]
            except (KeyError, AttributeError, TypeError, ValueError):
                return None
            airport_scope = f" for {airport_code} {airport_column.lower()} airport"

        try:
            monthly_stats = build_monthly_statistics(
                monthly_data,
                percentiles=sorted(percentile_numbers),
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return None

        if monthly_stats.empty:
            return {
                "answer": (
                    f"No cleaned flight records are available for monthly taxi-out analysis"
                    f"{airport_scope}."
                ),
                "sources": ["[data/processed/cleaned_flights.csv - monthly aggregation]"],
                "confidence": "High",
            }

        requested_years = sorted({int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", q_lower)})
        available_years = sorted(int(year) for year in monthly_stats["YEAR"].unique())
        if requested_years:
            requested_year = requested_years[-1]
            monthly_stats = monthly_stats.loc[monthly_stats["YEAR"] == requested_year]
            if monthly_stats.empty:
                available_text = ", ".join(str(year) for year in available_years)
                return {
                    "answer": (
                        f"No cleaned flight records are available for **{requested_year}**. "
                        f"Available dataset year(s): **{available_text}**."
                    ),
                    "sources": ["[data/processed/cleaned_flights.csv - available date range]"],
                    "confidence": "High",
                }
            title_suffix = (
                f"{airport_scope} in {requested_year}"
                if airport_scope
                else f" for {requested_year}"
            )
        else:
            title_suffix = airport_scope

        headers = ["Month"] + [spec[1] for spec in requested]
        lines = [f"**Monthly Taxi-Out Statistics{title_suffix}:**", ""]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in monthly_stats.sort_values(["YEAR", "MONTH_NUM"]).iterrows():
            period = f"{int(row['YEAR'])} {row['MONTH']}" if len(available_years) > 1 else row["MONTH"]
            values = [period] + [f"{row[spec[2]]:.1f}" for spec in requested]
            lines.append("| " + " | ".join(values) + " |")
        return {
            "answer": "\n".join(lines),
            "sources": ["[data/processed/cleaned_flights.csv - monthly taxi-out aggregation]"],
            "confidence": "High",
        }

    def _answer_from_airport_percentile(
        self,
        query: str,
        flight_data: Any,
    ) -> Optional[Dict[str, Any]]:
        """Answers a taxi-out percentile for an explicitly named airport."""
        q_lower = self._normalize_calculation_query(query)
        if any(term in q_lower for term in ["monthly", "per month", "by month", "each month"]):
            return None

        percentile_match = re.search(
            r"\bp[\s_-]?(\d{1,3})(?:st|nd|rd|th)?\b",
            q_lower,
        )
        if not percentile_match:
            percentile_match = re.search(
                r"\b(\d{1,3})(?:st|nd|rd|th)?\s+percentile\b",
                q_lower,
            )
        if not percentile_match:
            return None

        percentile = int(percentile_match.group(1))
        airports = self._find_airports(query, flight_data)
        if not airports or not 0 <= percentile <= 100:
            return None

        airport_column = "DEST" if any(
            term in q_lower for term in ["destination", "arrival"]
        ) else "ORIGIN"
        airport_label = "destination" if airport_column == "DEST" else "origin"
        lines = [f"**P{percentile} Taxi-Out by {airport_label.title()} Airport:**", ""]
        try:
            airport_values = flight_data[airport_column].astype(str).str.upper()
            taxi_values = pd.to_numeric(flight_data["TAXI_OUT"], errors="coerce")
        except (KeyError, AttributeError, TypeError, ValueError):
            return None

        for airport_code in airports:
            selected_values = taxi_values.loc[airport_values == airport_code].dropna()
            if selected_values.empty:
                lines.append(f"- **{airport_code}:** no cleaned flight records")
                continue
            percentile_value = selected_values.quantile(percentile / 100.0)
            lines.append(
                f"- **{airport_code}: {percentile_value:.1f} minutes** "
                f"({len(selected_values):,} cleaned flights)"
            )

        return {
            "answer": "\n".join(lines),
            "sources": ["[data/processed/cleaned_flights.csv - live airport percentile]"],
            "confidence": "High",
        }

    def _is_taxi_out_calculation_query(self, query: str, flight_data: Any) -> bool:
        """Infers taxi-out for scoped questions or safe dashboard-default time questions."""
        q_lower = self._normalize_calculation_query(query)
        calculation_terms = [
            "average", "avg", "mean", "median", "highest", "lowest", "largest",
            "smallest", "maximum", "minimum", "top ", "bottom ", "rank", "ranking",
        ]
        competing_terms = [
            "departure delay", "delay time", "flight duration", "flight time",
            "scheduled departure", "scheduled time", "distance", "model error",
            "mae", "rmse", "r-squared", "r2",
        ]
        has_percentile_intent = bool(
            re.search(r"\bp[\s_-]?\d{1,3}(?:st|nd|rd|th)?\b", q_lower)
            or re.search(r"\b\d{1,3}(?:st|nd|rd|th)?\s+percentile\b", q_lower)
        )
        has_calculation_intent = (
            any(term in q_lower for term in calculation_terms)
            or has_percentile_intent
        )
        has_explicit_taxi_out = "taxi" in q_lower or "ground time" in q_lower
        has_time_metric = (
            "time" in q_lower
            or "duration" in q_lower
            or has_explicit_taxi_out
            or has_percentile_intent
        )
        has_scope = any(
            term in q_lower
            for term in [
                "airport", "airports", "origin", "airline", "airlines", "carrier",
                "route", "routes", "departure hour", "hour of day",
            ]
        )
        if not has_scope and flight_data is not None:
            has_scope = bool(self._find_airports(query, flight_data) or self._find_carrier(query, flight_data))
        return (
            has_calculation_intent
            and has_time_metric
            and not any(term in q_lower for term in competing_terms)
            and (
                has_explicit_taxi_out
                or has_scope
                or (
                    not has_explicit_taxi_out
                    and not has_scope
                    and any(term in q_lower for term in ["average", "avg", "mean", "median", "how long"])
                )
            )
        )

    def _answer_from_overall_taxi_stats(
        self,
        query: str,
        flight_data: Any,
    ) -> Optional[Dict[str, Any]]:
        """Answers unscoped time questions using taxi-out as the dashboard metric."""
        q_lower = self._normalize_calculation_query(query)
        if not self._is_taxi_out_calculation_query(query, flight_data):
            return None
        if any(
            term in q_lower
            for term in [
                "airport", "airports", "origin", "destination", "arrival",
                "airline", "airlines", "carrier", "route", "routes",
                "departure hour", "hour of day",
            ]
        ):
            return None
        if not any(term in q_lower for term in ["average", "avg", "mean", "median", "how long"]):
            return None
        try:
            taxi_values = flight_data["TAXI_OUT"].dropna().astype(float)
        except (KeyError, AttributeError, TypeError, ValueError):
            return None
        if taxi_values.empty:
            return None
        return {
            "answer": (
                f"**Average Taxi-Out Time:** **{taxi_values.mean():.1f} minutes**\n\n"
                f"Based on **{len(taxi_values):,} cleaned flight records**. "
                f"The median taxi-out time is **{taxi_values.median():.1f} minutes**."
            ),
            "sources": ["[data/processed/cleaned_flights.csv - overall taxi-out aggregation]"],
            "confidence": "High",
        }
    def _answer_from_grouped_data(
        self,
        query: str,
        flight_data: Any,
        group_column: str,
        label: str,
    ) -> Optional[Dict[str, Any]]:
        """Answers ranking and scoped summary questions from cleaned flight records."""
        q_lower = self._normalize_calculation_query(query)
        try:
            values = flight_data[[group_column, "TAXI_OUT"]].copy()
            values["TAXI_OUT"] = values["TAXI_OUT"].astype(float)
            values = values.dropna(subset=[group_column, "TAXI_OUT"])
            summary = values.groupby(group_column)["TAXI_OUT"].agg(
                Mean="mean", Median="median", Flights="count"
            )
        except (KeyError, AttributeError, TypeError, ValueError):
            return None

        if summary.empty:
            return None

        is_highest = any(term in q_lower for term in ["highest", "largest", "maximum", "max"])
        is_lowest = any(term in q_lower for term in ["lowest", "smallest", "minimum", "min"])
        is_ranking = is_highest or is_lowest or any(
            term in q_lower for term in ["top ", "bottom ", "rank", "ranking"]
        )
        asks_taxi = self._is_taxi_out_calculation_query(query, flight_data)
        asks_average = any(term in q_lower for term in ["average", "mean", "median"])

        if is_ranking and asks_taxi:
            ascending = is_lowest and not is_highest
            default_limit = 1 if (is_highest or is_lowest) else 5
            ranked = summary.sort_values(
                "Mean", ascending=ascending
            ).head(self._requested_limit(query, default=default_limit))
            direction = "lowest" if ascending else "highest"
            lines = [f"**{direction.title()} {label} by average taxi-out:**", ""]
            for rank, (group_value, row) in enumerate(ranked.iterrows(), start=1):
                lines.append(
                    f"{rank}. **{group_value}** — **{row['Mean']:.1f} minutes** "
                    f"({int(row['Flights']):,} cleaned flights)"
                )
            return {
                "answer": "\n".join(lines),
                "sources": [f"[data/processed/cleaned_flights.csv - {label.lower()} taxi-out aggregation]"],
                "confidence": "High",
            }

        if asks_average and (label in ["Airport", "Departure Hour"] or self._find_airports(query, flight_data)):
            airports = self._find_airports(query, flight_data)
            if group_column == "ORIGIN" and airports:
                summary = summary.loc[summary.index.astype(str).str.upper().isin(airports)]
            if summary.empty:
                return None
            lines = [f"**Taxi-Out Summary by {label}:**", ""]
            for group_value, row in summary.sort_values("Mean", ascending=False).iterrows():
                lines.append(
                    f"- **{group_value}:** average **{row['Mean']:.1f} minutes**, "
                    f"median **{row['Median']:.1f} minutes** ({int(row['Flights']):,} cleaned flights)"
                )
            return {
                "answer": "\n".join(lines),
                "sources": [f"[data/processed/cleaned_flights.csv - {label.lower()} taxi-out aggregation]"],
                "confidence": "High",
            }

        return None

    def _answer_from_live_data(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        flight_data: Any
    ) -> Optional[Dict[str, Any]]:
        """Answers current-selection and route-level questions from the loaded flight data."""
        q_lower = self._normalize_calculation_query(query)
        context = context or {}
        selection_terms = ["current selection", "selected flight", "my selection", "current inputs"]
        selection_summary_terms = [
            "what is the current selection", "what are the current selections",
            "show current selection", "show my selection", "what is selected"
        ]

        if any(term in q_lower for term in selection_summary_terms):
            return {
                "answer": (
                    "**Current Dashboard Selection:**\n\n"
                    f"- Carrier: **{context.get('carrier', 'N/A')}**\n"
                    f"- Route: **{context.get('origin', 'N/A')} to {context.get('destination', 'N/A')}**\n"
                    f"- Flight date: **{context.get('flight_date', 'N/A')}**\n"
                    f"- Scheduled departure: **{context.get('scheduled_departure', 'N/A')}**\n"
                    f"- Departure delay: **{context.get('departure_delay', 'N/A')} minutes**\n"
                    f"- Distance: **{context.get('distance', 'N/A')} miles**"
                ),
                "sources": ["[Current dashboard selections]"],
                "confidence": "High"
            }

        monthly_route_answer = self._answer_from_monthly_route_data(query, flight_data) if flight_data is not None else None
        if monthly_route_answer:
            return monthly_route_answer

        monthly_answer = self._answer_from_monthly_data(query, flight_data) if flight_data is not None else None
        if monthly_answer:
            return monthly_answer

        airport_percentile_answer = self._answer_from_airport_percentile(
            query,
            flight_data,
        ) if flight_data is not None else None
        if airport_percentile_answer:
            return airport_percentile_answer

        generic_answer = self._answer_from_generic_aggregation(query, flight_data) if flight_data is not None else None
        if generic_answer:
            return generic_answer

        route_answer = self._answer_from_route_data(query, flight_data) if flight_data is not None else None
        if route_answer:
            return route_answer

        distance_answer = self._answer_from_distance_data(query, flight_data) if flight_data is not None else None
        if distance_answer:
            return distance_answer

        record_share_answer = self._answer_from_record_share(query, flight_data) if flight_data is not None else None
        if record_share_answer:
            return record_share_answer

        record_count_answer = self._answer_from_record_count_ranking(query, flight_data) if flight_data is not None else None
        if record_count_answer:
            return record_count_answer

        # Ranking questions must be answered from the records, not from the markdown index.
        if flight_data is not None and self._is_taxi_out_calculation_query(query, flight_data):
            mentions_airport = any(term in q_lower for term in ["airport", "airports", "origin"])
            mentions_carrier = any(term in q_lower for term in ["airline", "airlines", "carrier"])
            mentions_hour = any(term in q_lower for term in ["departure hour", "departure time", "hour of day", "time of day"])
            if mentions_airport:
                grouped_answer = self._answer_from_grouped_data(
                    query, flight_data, "ORIGIN", "Airport"
                )
                if grouped_answer:
                    return grouped_answer
            if mentions_carrier:
                grouped_answer = self._answer_from_grouped_data(
                    query, flight_data, "OP_UNIQUE_CARRIER", "Carrier"
                )
                if grouped_answer:
                    return grouped_answer
            if mentions_hour and "DEP_HOUR" in flight_data.columns:
                grouped_answer = self._answer_from_grouped_data(
                    query, flight_data, "DEP_HOUR", "Departure Hour"
                )
                if grouped_answer:
                    return grouped_answer
        overall_answer = self._answer_from_overall_taxi_stats(query, flight_data)
        if overall_answer:
            return overall_answer

        time_question = any(term in q_lower for term in ["average", "mean", "median", "how long"]) and (
            "taxi" in q_lower or "time" in q_lower
        )
        if not time_question or flight_data is None:
            return None

        carrier_code = self._find_carrier(query, flight_data)
        airports = self._find_airports(query, flight_data)
        uses_selection = any(
            term in q_lower
            for term in [*selection_terms, "current route", "selected route", "this route"]
        )

        if carrier_code and len(airports) < 2 and not uses_selection:
            try:
                carrier_mask = (
                    flight_data["OP_UNIQUE_CARRIER"].astype(str).str.upper() == carrier_code
                )
                taxi_values = flight_data.loc[carrier_mask, "TAXI_OUT"].dropna()
            except (KeyError, AttributeError, TypeError):
                return None

            if taxi_values.empty:
                return {
                    "answer": f"No cleaned flight records were found for carrier **{carrier_code}**.",
                    "sources": ["[data/processed/cleaned_flights.csv - live carrier filter]"],
                    "confidence": "High"
                }
            return {
                "answer": (
                    f"**Average Taxi-Out Time for {carrier_code}: {taxi_values.mean():.1f} minutes**\n\n"
                    f"Based on **{len(taxi_values):,} cleaned flight records**. "
                    f"The median taxi-out time is **{taxi_values.median():.1f} minutes**."
                ),
                "sources": ["[data/processed/cleaned_flights.csv - live carrier filter]"],
                "confidence": "High"
            }

        if len(airports) == 1 and not uses_selection:
            airport_code = airports[0]
            try:
                airport_mask = flight_data["ORIGIN"].astype(str).str.upper() == airport_code
                if carrier_code:
                    airport_mask &= (
                        flight_data["OP_UNIQUE_CARRIER"].astype(str).str.upper() == carrier_code
                    )
                taxi_values = flight_data.loc[airport_mask, "TAXI_OUT"].dropna()
            except (KeyError, AttributeError, TypeError):
                return None

            scope = f" for carrier {carrier_code}" if carrier_code else ""
            if taxi_values.empty:
                return {
                    "answer": f"No cleaned flight records were found for origin airport **{airport_code}**{scope}.",
                    "sources": ["[data/processed/cleaned_flights.csv - live airport filter]"],
                    "confidence": "High",
                }
            return {
                "answer": (
                    f"**Average Taxi-Out Time for {airport_code}: {taxi_values.mean():.1f} minutes**\n\n"
                    f"Based on **{len(taxi_values):,} cleaned flight records**{scope}. "
                    f"The median taxi-out time is **{taxi_values.median():.1f} minutes**."
                ),
                "sources": ["[data/processed/cleaned_flights.csv - live airport filter]"],
                "confidence": "High",
            }
        if len(airports) >= 2:
            origin_code, destination_code = airports[:2]
        elif uses_selection and context.get("origin") and context.get("destination"):
            origin_code = str(context["origin"]).upper()
            destination_code = str(context["destination"]).upper()
        else:
            return None

        try:
            origins = flight_data["ORIGIN"].astype(str).str.upper()
            destinations = flight_data["DEST"].astype(str).str.upper()
            bidirectional = "between" in q_lower and " and " in q_lower
            if bidirectional:
                route_mask = (
                    ((origins == origin_code) & (destinations == destination_code))
                    | ((origins == destination_code) & (destinations == origin_code))
                )
                route_label = f"{origin_code} and {destination_code} (both directions)"
            else:
                route_mask = (origins == origin_code) & (destinations == destination_code)
                route_label = f"{origin_code} to {destination_code}"

            if (
                carrier_code is None
                and any(term in q_lower for term in selection_terms)
                and context.get("carrier")
            ):
                carrier_code = str(context["carrier"]).upper()
            if carrier_code:
                route_mask &= (
                    flight_data["OP_UNIQUE_CARRIER"].astype(str).str.upper() == carrier_code
                )

            taxi_values = flight_data.loc[route_mask, "TAXI_OUT"].dropna()
        except (KeyError, AttributeError, TypeError):
            return None

        scope = f" for carrier {carrier_code}" if carrier_code else ""
        if taxi_values.empty:
            return {
                "answer": f"No cleaned flight records were found for **{route_label}**{scope}.",
                "sources": ["[data/processed/cleaned_flights.csv - live route filter]"],
                "confidence": "High"
            }

        return {
            "answer": (
                f"**Average Taxi-Out Time - {route_label}: {taxi_values.mean():.1f} minutes**\n\n"
                f"Based on **{len(taxi_values):,} cleaned flight records**{scope}. "
                f"The median taxi-out time is **{taxi_values.median():.1f} minutes**."
            ),
            "sources": ["[data/processed/cleaned_flights.csv - live route filter]"],
            "confidence": "High"
        }

    def _answer_from_dashboard_guide(self, query: str) -> Optional[Dict[str, Any]]:
        """Returns the curated interpretation for a named dashboard chart or value."""
        normalized_query = re.sub(
            r"[^a-z0-9]+",
            " ",
            self._normalize_calculation_query(query),
        ).strip()
        explanation_terms = [
            "what is", "what does", "what do", "what are", "explain",
            "interpret", "meaning", "denote", "define", "describe",
            "how to read", "tell me", "tell us",
        ]
        if not any(term in normalized_query for term in explanation_terms):
            return None

        topic_aliases = {
            "pushback to airborne sequence breakdown": "Pushback-to-Airborne Sequence Breakdown",
            "current prediction queue contribution impact versus low congestion operational status": "Current prediction, Queue contribution, Impact versus low congestion, and Operational status",
            "congestion heatmap by day and departure hour": "Congestion Heatmap by Day and Departure Hour",
            "exploratory mean median 95th percentile above 30 minutes": "Exploratory mean, median, 95th percentile, and Above 30 minutes",
            "live congestion sensitivity curve": "Live Congestion Sensitivity Curve",
            "hub airport average taxi out times": "Hub Airport Average Taxi-Out Times",
            "diurnal departure bank wave profile": "Diurnal Departure Bank Wave Profile",
            "flight distance versus departure delay": "Flight Distance versus Departure Delay",
            "departure delay versus taxi out": "Departure Delay versus Taxi-Out",
            "departure delay by distance band": "Departure Delay by Distance Band",
            "mean taxi out by departure hour": "Mean Taxi-Out by Departure Hour",
            "top feature importances for random forest": "Top Feature Importances for Random Forest",
            "highest mean taxi out origins": "Highest Mean Taxi-Out Origins",
            "dispatch decision support scenario chart": "Dispatch Decision Support scenario chart",
            "operational risk confusion matrix": "Operational Risk Confusion Matrix",
            "random forest benchmark cards": "Random Forest benchmark cards",
            "taxi out target reference": "Taxi-Out Target Reference",
            "carrier mean taxi out": "Carrier Mean Taxi-Out",
            "monthly taxi out trend": "Monthly Taxi-Out Trend",
            "high risk departures by hour": "High-Risk Departures by Hour",
            "origin airport comparison": "Origin Airport Comparison",
            "comparative model scores": "Comparative Model Scores",
            "model error comparison": "Model error comparison",
            "taxi out distribution": "Taxi-Out Distribution",
            "data cleaning funnel": "Data Cleaning Funnel",
            "estimated wheels off": "Estimated Wheels-Off",
            "actual pushback and scheduled departure": "Actual Pushback and Scheduled Departure",
            "scheduled departure": "Actual Pushback and Scheduled Departure",
            "actual pushback": "Actual Pushback and Scheduled Departure",
            "predicted taxi out": "Predicted Taxi-Out",
            "surface congestion": "Surface Congestion",
            "current prediction": "Current prediction, Queue contribution, Impact versus low congestion, and Operational status",
            "impact versus low congestion": "Current prediction, Queue contribution, Impact versus low congestion, and Operational status",
            "queue contribution": "Current prediction, Queue contribution, Impact versus low congestion, and Operational status",
            "operational status": "Current prediction, Queue contribution, Impact versus low congestion, and Operational status",
            "random forest rmse": "Random Forest benchmark cards",
            "random forest mae": "Random Forest benchmark cards",
            "within 3 min tolerance": "Random Forest benchmark cards",
            "within 5 min tolerance": "Random Forest benchmark cards",
            "model r2 score": "Random Forest benchmark cards",
            "busiest routes": "Busiest Routes",
            "flights analysed": "Flights analysed",
            "flights analyzed": "Flights analysed",
            "median taxi out": "Exploratory mean, median, 95th percentile, and Above 30 minutes",
            "mean taxi out": "Mean Taxi-Out",
            "retention rate": "Retention Rate",
            "clean records": "Clean Records",
            "raw records": "Raw Records",
            "95th percentile": "Exploratory mean, median, 95th percentile, and Above 30 minutes",
            "above 30 minutes": "Exploratory mean, median, 95th percentile, and Above 30 minutes",
            "confusion matrix": "Operational Risk Confusion Matrix",
        }
        matched_section = next(
            (
                section
                for alias, section in sorted(
                    topic_aliases.items(),
                    key=lambda item: len(item[0]),
                    reverse=True,
                )
                if alias in normalized_query
            ),
            None,
        )
        if not matched_section:
            return None

        section_chunks = [
            chunk["text"].strip()
            for chunk in self.chunks
            if chunk["file_path"] == os.path.join("docs", "ai_analyst_knowledge.md")
            and chunk["section"] == matched_section
        ]
        if not section_chunks:
            return None

        return {
            "answer": "\n\n".join(section_chunks[:3]),
            "sources": [f"[docs/ai_analyst_knowledge.md - {matched_section}]"],
            "confidence": "High",
            "topic": matched_section,
        }

    def _resolve_contextual_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Adds the prior dashboard topic only when the wording is referential."""
        context = context or {}
        normalized_query = self._normalize_calculation_query(query)
        reference_terms = [
            "this chart", "this graph", "this plot", "this table",
            "in this", "which has", "which is", "which one", "that chart",
            "that graph", "above chart", "above graph", "these values",
            "what does it", "what is it", "what is the maximum",
            "what is the minimum", "tell me more", "explain further",
            "more detail", "go deeper",
        ]
        if not any(term in normalized_query for term in reference_terms):
            return query

        topic = context.get("last_dashboard_topic")
        if topic:
            return f"{query} Dashboard topic: {topic}."

        previous_query = context.get("previous_user_query")
        if previous_query:
            return f"{query} Previous analytical question: {previous_query}"
        return query

    def _answer_from_dashboard_follow_up(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Answers extrema follow-ups for dashboard charts with stored values."""
        context = context or {}
        normalized_query = self._normalize_calculation_query(query)
        reference_terms = [
            "this chart", "this graph", "this plot", "this table",
            "in this", "which has", "which is", "which one", "that chart",
            "that graph", "above chart", "above graph",
        ]
        query_words = set(re.findall(r"\b[a-z0-9]+\b", normalized_query))
        asks_maximum = bool(
            query_words & {"max", "maximum", "highest", "largest", "most"}
        )
        asks_minimum = bool(
            query_words & {"min", "minimum", "lowest", "smallest", "least"}
        )
        if (
            not any(term in normalized_query for term in reference_terms)
            or not (asks_maximum or asks_minimum)
        ):
            return None

        topic = str(context.get("last_dashboard_topic") or "")
        previous_query = str(context.get("previous_user_query") or "")
        topic_search = f"{topic} {previous_query}".lower()

        if "data cleaning funnel" in topic_search:
            raw_count = int(self.audit_data.get("raw_record_count", 75000))
            cancelled = int(self.audit_data.get("cancelled_flights_removed", 1507))
            diverted = int(self.audit_data.get("diverted_flights_removed", 194))
            invalid = int(self.audit_data.get("invalid_or_missing_taxi_out_removed", 438))
            outliers = int(self.audit_data.get("outliers_removed", 589))
            final_count = int(self.audit_data.get("final_cleaned_count", 72272))
            asks_change = any(
                term in normalized_query
                for term in ["drop", "reduction", "removed", "removal", "decrease"]
            )

            if asks_change:
                removals = {
                    "Cancelled-flight removal": cancelled,
                    "Diverted-flight removal": diverted,
                    "Invalid-target removal": invalid,
                    "Outlier removal": outliers,
                }
                label, value = (
                    max(removals.items(), key=lambda item: item[1])
                    if asks_maximum
                    else min(removals.items(), key=lambda item: item[1])
                )
                answer = (
                    f"**{label} is the {'largest' if asks_maximum else 'smallest'} drop: "
                    f"{value:,} records.**"
                )
            else:
                answer = (
                    f"**The maximum bar is Raw records: {raw_count:,}.**\n\n"
                    f"It is the starting dataset before any filters. The minimum bar is "
                    f"Final cleaned records: **{final_count:,}**. If you meant the largest "
                    f"single reduction, it is cancelled-flight removal: **{cancelled:,} records**."
                    if asks_maximum
                    else (
                        f"**The minimum bar is Final cleaned records: {final_count:,}.**\n\n"
                        f"It is the remaining dataset after all filters. The maximum bar is "
                        f"Raw records: **{raw_count:,}**."
                    )
                )
            return {
                "answer": answer,
                "sources": ["[data/processed/cleaning_audit_summary.json - cleaning stages]"],
                "confidence": "High",
                "topic": "Data Cleaning Funnel",
            }

        static_extrema = {
            "taxi-out target reference": {
                "max": "Maximum taxi-out is 119.0 minutes.",
                "min": "Minimum taxi-out is 11.7 minutes.",
            },
            "hub airport average taxi-out times": {
                "max": "JFK has the highest configured reference average at 28.4 minutes.",
                "min": "DEN has the lowest configured reference average at 16.2 minutes.",
            },
            "diurnal departure bank wave profile": {
                "max": "Hour 17 has the highest configured reference value at 24.1 minutes.",
                "min": "Hour 2 has the lowest configured reference value at 13.5 minutes.",
            },
            "top feature importances for random forest": {
                "max": "DISTANCE has the highest Random Forest importance at 30.03 percent.",
                "min": "Among the displayed top ten, DAY_OF_WEEK_4 has the lowest importance at 1.97 percent.",
            },
            "model error comparison": {
                "max": "Linear Regression RMSE is the largest displayed error bar at 10.037 minutes.",
                "min": "XGBoost MAE is the smallest displayed error bar at 3.879 minutes.",
            },
        }
        for chart_name, extrema in static_extrema.items():
            if chart_name in topic_search:
                direction = "max" if asks_maximum else "min"
                return {
                    "answer": f"**{extrema[direction]}**",
                    "sources": [f"[docs/ai_analyst_knowledge.md - {topic or chart_name}]"],
                    "confidence": "High",
                    "topic": topic or chart_name,
                }
        return None

    def build_index(self):
        """Indexes all markdown project files."""
        docs = load_markdown_documents(self.base_dir)
        source_weights = {
            os.path.join("docs", "ai_analyst_knowledge.md"): 1.40,
            "Project_Report.md": 1.15,
            os.path.join("docs", "architecture.md"): 1.10,
            os.path.join("docs", "data_dictionary.md"): 1.10,
            os.path.join("docs", "user_guide.md"): 1.00,
            "README.md": 0.90,
            "PARAMETERS.md": 0.85,
            os.path.join("output", "pdf", "2606MGAIL130_Abstract.md"): 0.75,
            "IMPROVEMENTS.md": 0.45,
        }
        all_chunks = []
        for doc in docs:
            chunks = chunk_markdown(doc)
            for chunk in chunks:
                chunk["source_weight"] = source_weights.get(chunk["file_path"], 1.0)
            all_chunks.extend(chunks)

        if not all_chunks:
            self.is_indexed = False
            return

        self.chunks = all_chunks
        corpus = [f"{c['section']}. {c['text']}" for c in self.chunks]
        
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.is_indexed = True

    def _extract_direct_sentences(self, query: str, text: str, max_sentences: int = 3) -> str:
        """
        Extracts only the most relevant sentences matching query terms from a chunk,
        avoiding dumping massive paragraphs.
        """
        # If it's a markdown table or list, keep it clean
        if "|" in text:
            return text.strip()
            
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= max_sentences:
            return text.strip()

        query_words = set(re.findall(r'\w+', query.lower())) - {"how", "many", "what", "is", "are", "you", "have", "the", "in", "here", "this"}
        
        scored_sentences = []
        for i, s in enumerate(sentences):
            s_words = set(re.findall(r'\w+', s.lower()))
            overlap = len(query_words & s_words)
            scored_sentences.append((overlap, i, s.strip()))

        # Sort by relevance, then chronological order
        scored_sentences.sort(key=lambda x: (-x[0], x[1]))
        top_sentences = sorted(scored_sentences[:max_sentences], key=lambda x: x[1])
        
        result = " ".join([s[2] for s in top_sentences if s[0] > 0])
        return result if result else sentences[0]

    def answer_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        flight_data: Any = None
    ) -> Dict[str, Any]:
        """
        Returns a focused, direct answer to the user's question without dumping surrounding noise.
        """
        context = context or {}
        follow_up_answer = self._answer_from_dashboard_follow_up(query, context)
        if follow_up_answer:
            return follow_up_answer

        effective_query = self._resolve_contextual_query(query, context)
        q_lower = self._normalize_calculation_query(effective_query)

        if (
            self._sources_requested(query)
            and not self._has_project_context(query)
            and len(re.findall(r"\w+", query)) <= 7
        ):
            previous_sources = context.get("previous_sources", [])
            if previous_sources:
                return {
                    "answer": "The supporting sources for the previous answer are available below.",
                    "sources": previous_sources,
                    "confidence": "High",
                }
            return {
                "answer": "Please identify the answer or project topic for which you want sources.",
                "sources": [],
                "confidence": "Low",
            }

        if not self._has_project_context(effective_query):
            return {
                "answer": (
                    "Please add a project topic and a specific request so I can answer it as a clear question. "
                    "For example: **What is the average taxi-out time?** or "
                    "**How does congestion affect the prediction?**"
                ),
                "sources": [],
                "confidence": "Low",
            }

        prefer_live_answer = (
            flight_data is not None
            and self._is_taxi_out_calculation_query(effective_query, flight_data)
        )
        if prefer_live_answer:
            live_answer = self._answer_from_live_data(effective_query, context, flight_data)
            if live_answer:
                return live_answer

        dashboard_guide_answer = self._answer_from_dashboard_guide(effective_query)
        if dashboard_guide_answer:
            return dashboard_guide_answer

        if not prefer_live_answer:
            live_answer = self._answer_from_live_data(effective_query, context, flight_data)
            if live_answer:
                return live_answer

        model_answer = self._answer_from_model_knowledge(effective_query)
        if model_answer:
            return model_answer

        workbook_answer = self._answer_from_excel(q_lower)
        if workbook_answer:
            return workbook_answer

        if any(term in q_lower for term in ["mean taxi", "average taxi", "average taxi time"]):
            stats = self.audit_data.get("post_clean_stats", {})
            mean_taxi_out = stats.get("mean_taxi_out")
            median_taxi_out = stats.get("median_taxi_out")
            if mean_taxi_out is not None:
                median_text = f" The median is {median_taxi_out:.1f} minutes." if median_taxi_out is not None else ""
                return {
                    "answer": (
                        f"**Mean Taxi-Out Time:** {mean_taxi_out:.1f} minutes\n\n"
                        f"This is the average time from aircraft pushback to wheels-off across the cleaned flight dataset."
                        f"{median_text}"
                    ),
                    "sources": ["[data/processed/cleaning_audit_summary.json - post_clean_stats]", "[Project_Report.md - Data Cleaning & Retention]"],
                    "confidence": "High"
                }

        # Direct factual handlers for high precision
        if any(w in q_lower for w in ["how many record", "number of record", "total record", "raw record", "cleaned record", "cleared record"]):
            raw_cnt = self.audit_data.get("raw_record_count", 75000)
            clean_cnt = self.audit_data.get("final_cleaned_count", 72272)
            retention = self.audit_data.get("retention_rate_pct", "96.36")
            excluded_cnt = raw_cnt - clean_cnt
            
            return {
                "answer": (
                    f"**Dataset Record Counts:**\n\n"
                    f"- **Raw Records:** {raw_cnt:,} flights\n"
                    f"- **Clean Records (Post-Cleaning):** {clean_cnt:,} flights\n"
                    f"- **Data Retention Rate:** {retention}%\n"
                    f"- **Excluded Records:** {excluded_cnt:,} flights (cancelled, diverted, missing taxi values, and >120 min outliers)."
                ),
                "sources": ["[data/processed/cleaning_audit_summary.json]", "[docs/ai_analyst_knowledge.md - Data cleaning and quality controls]"],
                "confidence": "High"
            }

        # Vector retrieval for general queries
        if not self.is_indexed or not self.vectorizer or self.tfidf_matrix is None:
            return {
                "answer": "Documentation index is not ready.",
                "sources": [],
                "confidence": "Low"
            }

        query_vec = self.vectorizer.transform([effective_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        source_weights = np.array([chunk.get("source_weight", 1.0) for chunk in self.chunks])
        if any(term in q_lower for term in ["improvement", "improve", "roadmap", "future work", "todo"]):
            source_weights = np.array([
                1.05 if chunk["file_path"] == "IMPROVEMENTS.md" else chunk.get("source_weight", 1.0)
                for chunk in self.chunks
            ])
        adjusted_similarities = similarities * source_weights
        top_indices = np.argsort(adjusted_similarities)[::-1][:3]

        results = []
        for idx in top_indices:
            score = float(adjusted_similarities[idx])
            if score >= 0.12:
                results.append((score, self.chunks[idx]))

        if not results:
            return {
                "answer": "I could not find a sufficiently reliable answer in the project knowledge sources. Please add the missing topic to the authoritative AI Analyst knowledge document or ask for a more specific project metric.",
                "sources": [],
                "confidence": "Low"
            }

        # Extract only the pertinent sentences from the best chunk
        best_score, best_chunk = results[0]
        focused_text = self._extract_direct_sentences(effective_query, best_chunk["text"], max_sentences=3)
        sources = [f"[{best_chunk['file_path']} - {best_chunk['section']}]"]

        return {
            "answer": focused_text,
            "sources": sources,
            "confidence": "High" if best_score >= 0.25 else "Moderate"
        }
