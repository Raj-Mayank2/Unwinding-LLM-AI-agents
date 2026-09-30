import os
import json
import hashlib
from typing import Literal, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent


# ============================================================
# Setup
# ============================================================

load_dotenv()

st.set_page_config(page_title="Ask My Dataset", page_icon="📊", layout="wide")

st.title("📊 Ask My Dataset")
st.caption("AI Data Analyst powered by LangGraph + Groq")

# llama-3.3-70b-versatile first: it is the most reliable at tool calling
MODEL_OPTIONS = [
    "openai/gpt-oss-20b",
]


# ============================================================
# Helpers (safe across Streamlit versions)
# ============================================================

def make_arrow_safe(data: pd.DataFrame) -> pd.DataFrame:
    """Make a dataframe safe for st.dataframe (PyArrow) rendering."""
    safe = data.copy()
    safe.columns = [str(c) for c in safe.columns]
    for col in safe.columns:
        if safe[col].dtype == "object":
            # Mixed-type columns (numbers + text) break Arrow serialization
            safe[col] = safe[col].map(lambda v: "" if pd.isna(v) else str(v))
    return safe


def show_df(data: pd.DataFrame, height=None):
    """Display a dataframe safely across Streamlit versions."""
    if data is None or data.empty:
        st.warning("This table has no rows to display.")
        return

    safe = make_arrow_safe(data)

    # Only pass height when explicitly set (None is invalid in new Streamlit)
    kwargs = {}
    if height is not None:
        kwargs["height"] = height

    try:
        st.dataframe(safe, width="stretch", **kwargs)  # newer Streamlit
    except Exception:
        try:
            st.dataframe(safe, use_container_width=True, **kwargs)  # older Streamlit
        except Exception:
            st.table(safe.head(50))  # plain table, no Arrow needed


def show_chart(fig, key=None):
    """Display a Plotly chart safely across Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch", key=key)  # newer Streamlit
    except Exception:
        st.plotly_chart(fig, use_container_width=True, key=key)  # older Streamlit


def load_csv(uploaded) -> pd.DataFrame:
    """Read a CSV, trying several encodings and auto-detecting the delimiter."""
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            uploaded.seek(0)
            return pd.read_csv(uploaded, encoding=encoding, sep=None, engine="python")
        except UnicodeDecodeError as e:
            last_error = e
        except Exception:
            # Delimiter sniffing failed: fall back to the default comma parser
            try:
                uploaded.seek(0)
                return pd.read_csv(uploaded, encoding=encoding)
            except UnicodeDecodeError as e2:
                last_error = e2
            except Exception as e2:
                last_error = e2
                break
    raise last_error


def clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Strip column names and convert numeric-looking text columns to numbers."""
    data = data.copy()
    data.columns = [str(c).strip() for c in data.columns]

    null_like = {"", "nan", "NaN", "None", "null", "NULL", "N/A", "n/a", "NA", "-"}
    for col in data.columns:
        if data[col].dtype == "object":
            s = data[col].astype(str).str.strip()
            s = s.where(~s.isin(null_like))  # null-like text becomes NaN
            numeric = pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")
            non_null = s.notna().sum()
            # Convert only if at least 80% of the non-empty values are numeric
            if non_null > 0 and numeric.notna().sum() / non_null >= 0.8:
                data[col] = numeric
    return data


def file_signature(uploaded) -> str:
    return hashlib.md5(uploaded.getvalue()).hexdigest()


def message_text(content) -> str:
    """Normalize LLM message content (string or list of blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return str(content)


# ============================================================
# Tool factory (tools are bound to a specific dataframe)
# ============================================================

def build_agent(df: pd.DataFrame, model_name: str, charts: list):
    """Create an agent whose tools operate on the given dataframe."""

    def col_error(column: str) -> str:
        return (
            f"Column '{column}' does not exist. "
            f"Available columns: {list(df.columns)}"
        )

    @tool
    def inspect_dataset() -> str:
        """Inspect the dataset structure: rows, columns, data types, missing values."""
        info = {
            "rows": int(df.shape[0]),
            "columns": list(df.columns),
            "data_types": {c: str(t) for c, t in df.dtypes.items()},
            "missing_values": {c: int(v) for c, v in df.isna().sum().items()},
            "sample_rows": json.loads(
                df.head(3).to_json(orient="records", date_format="iso")
            ),
        }
        return json.dumps(info, indent=2, default=str)

    @tool
    def get_statistics(column: str) -> str:
        """Descriptive statistics (count, mean, std, min, quartiles, max) for a numeric column."""
        if column not in df.columns:
            return col_error(column)
        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"Column '{column}' is not numeric."
        if df[column].notna().sum() == 0:
            return f"Column '{column}' has no values: all {len(df)} rows are missing."
        return json.dumps(df[column].describe().to_dict(), indent=2, default=str)

    @tool
    def get_extreme_row(column: str, mode: Literal["max", "min"] = "max") -> str:
        """Find the highest (max) or lowest (min) value in a numeric column and
        return the full row(s) holding it, so other columns like gender or
        region can be read from the result."""
        if column not in df.columns:
            return col_error(column)
        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"Column '{column}' is not numeric."
        series = df[column]
        if series.notna().sum() == 0:
            return f"Column '{column}' has no values."
        target = series.max() if mode == "max" else series.min()
        if hasattr(target, "item"):
            target = target.item()
        matches = df[series == target]
        return json.dumps(
            {
                "column": column,
                "mode": mode,
                "value": target,
                "match_count": int(len(matches)),
                "rows": json.loads(
                    matches.head(5).to_json(orient="records", date_format="iso")
                ),
            },
            indent=2,
            default=str,
        )

    @tool
    def analyze_groups(
        group_column: str,
        metric_column: str,
        aggregation: Literal["sum", "mean", "median", "min", "max", "count"] = "sum",
        top_n: int = 20,
    ) -> str:
        """Group by group_column and aggregate metric_column
        (sum, mean, median, min, max, count). Returns results sorted descending."""
        try:
            if group_column not in df.columns:
                return col_error(group_column)
            if metric_column not in df.columns:
                return col_error(metric_column)
            if aggregation != "count" and not pd.api.types.is_numeric_dtype(
                df[metric_column]
            ):
                return f"Column '{metric_column}' must be numeric for '{aggregation}'."

            result = (
                df.groupby(group_column)[metric_column]
                .agg(aggregation)
                .sort_values(ascending=False)
                .head(top_n)
            )
            return result.to_json()
        except Exception as e:
            return f"Error: {e}"

    @tool
    def get_unique_values(column: str) -> str:
        """Get unique values (max 100) from a column."""
        if column not in df.columns:
            return col_error(column)
        values = df[column].dropna().unique().tolist()
        return json.dumps(
            {"total_unique": len(values), "values": values[:100]},
            indent=2,
            default=str,
        )

    @tool
    def value_counts(column: str, top_n: int = 15) -> str:
        """Count how often each value appears in a column."""
        if column not in df.columns:
            return col_error(column)
        return df[column].value_counts().head(top_n).to_json()

    @tool
    def filter_rows(
        column: str,
        operator: Literal["==", "!=", ">", "<", ">=", "<=", "contains"],
        value: str,
    ) -> str:
        """Filter rows where `column` <operator> `value`. Returns the match count
        and the first 10 matching rows."""
        limit = 10
        try:
            if column not in df.columns:
                return col_error(column)

            series = df[column]
            if operator == "contains":
                mask = series.astype(str).str.contains(value, case=False, na=False)
            else:
                if pd.api.types.is_numeric_dtype(series):
                    value_cast = float(value)
                elif pd.api.types.is_datetime64_any_dtype(series):
                    value_cast = pd.to_datetime(value)
                else:
                    value_cast = value
                ops = {
                    "==": lambda: series == value_cast,
                    "!=": lambda: series != value_cast,
                    ">": lambda: series > value_cast,
                    "<": lambda: series < value_cast,
                    ">=": lambda: series >= value_cast,
                    "<=": lambda: series <= value_cast,
                }
                mask = ops[operator]()

            matches = df[mask]
            return json.dumps(
                {
                    "match_count": int(len(matches)),
                    "rows": json.loads(
                        matches.head(limit).to_json(
                            orient="records", date_format="iso"
                        )
                    ),
                },
                indent=2,
                default=str,
            )
        except Exception as e:
            return f"Error: {e}"

    @tool
    def correlation(column_a: str, column_b: str) -> str:
        """Pearson correlation between two numeric columns."""
        for c in (column_a, column_b):
            if c not in df.columns:
                return col_error(c)
            if not pd.api.types.is_numeric_dtype(df[c]):
                return f"Column '{c}' is not numeric."
        return json.dumps({"correlation": float(df[column_a].corr(df[column_b]))})

    @tool
    def create_chart(
        chart_type: Literal["bar", "line", "pie", "scatter", "histogram", "box"],
        x: str,
        y: Optional[str] = None,
        aggregation: Literal["sum", "mean", "median", "min", "max", "count"] = "sum",
        title: str = "",
    ) -> str:
        """Create a chart shown to the user. For bar/line/pie: x is the category
        (or time) column and y an optional numeric column aggregated per x
        (if y is omitted, rows are counted). For scatter: x and y are both numeric.
        For histogram/box: x is a numeric column (y optional for box grouping)."""
        try:
            if x not in df.columns:
                return col_error(x)
            if y is not None and y not in df.columns:
                return col_error(y)

            chart_title = title or f"{chart_type.title()} chart of {y or x}"

            if chart_type in ("bar", "line", "pie"):
                if y:
                    data = df.groupby(x)[y].agg(aggregation).reset_index()
                    value_col = y
                else:
                    data = df[x].value_counts().reset_index()
                    data.columns = [x, "count"]
                    value_col = "count"

                if chart_type == "line":
                    data = data.sort_values(x)
                    fig = px.line(
                        data, x=x, y=value_col, title=chart_title, markers=True
                    )
                else:
                    data = data.sort_values(value_col, ascending=False).head(20)
                    if chart_type == "bar":
                        fig = px.bar(data, x=x, y=value_col, title=chart_title)
                    else:
                        fig = px.pie(
                            data, names=x, values=value_col, title=chart_title
                        )

            elif chart_type == "scatter":
                if not y:
                    return "Scatter charts need both x and y."
                fig = px.scatter(df, x=x, y=y, title=chart_title)

            elif chart_type == "histogram":
                fig = px.histogram(df, x=x, title=chart_title)

            else:  # box
                fig = px.box(df, x=x if y else None, y=y or x, title=chart_title)

            charts.append(fig)
            return "Chart created and displayed to the user."
        except Exception as e:
            return f"Error creating chart: {e}"

    llm = ChatGroq(model=model_name, temperature=0)

    system_prompt = """
You are an expert AI Data Analyst answering questions about an uploaded CSV dataset.

RULES:
1. Always use tools when the answer requires information from the dataset.
2. Never invent numbers. Every figure must come from a tool result.
3. If unsure about column names or types, call inspect_dataset first.
4. Use get_statistics for numeric summaries, analyze_groups for comparing
   categories, value_counts / get_unique_values for categorical columns,
   filter_rows for specific records, and correlation for relationships.
5. When the user asks for a chart, plot, graph or visualization,
   call create_chart.
6. If a requested column does not exist, list the available columns and
   suggest the closest match.
7. Give a clear, concise answer, then add one short insight when useful.
8. Never modify the dataset.
9. For "highest/lowest/oldest/youngest ... and its <other column>" questions,
   call get_extreme_row instead of filter_rows. If several rows tie for the
   extreme value, report all of them.
10. Call only one tool at a time and keep tool arguments minimal.
"""

    return create_react_agent(
        model=llm,
        tools=[
            inspect_dataset,
            get_statistics,
            get_extreme_row,
            analyze_groups,
            get_unique_values,
            value_counts,
            filter_rows,
            correlation,
            create_chart,
        ],
        prompt=system_prompt,
    )


# ============================================================
# Session State
# ============================================================

defaults = {
    "df": None,
    "file_sig": None,
    "agent": None,
    "agent_key": None,
    "charts": [],
    "chat": [],  # list of {"role", "content", "charts"}
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")

    model_name = st.selectbox(
        "Groq model",
        MODEL_OPTIONS,
        help="If a model keeps failing on tool calls, try another one.",
    )

    if os.getenv("GROQ_API_KEY"):
        st.success("GROQ_API_KEY found")
    else:
        st.error("GROQ_API_KEY missing. Add it to your .env file.")

    st.divider()
    st.subheader("💡 Try asking")
    st.markdown(
        "- What is the max age and its gender?\n"
        "- Which region has the highest total revenue?\n"
        "- Show a bar chart of revenue by category\n"
        "- Are Age and Revenue correlated?\n"
        "- Show rows where Revenue > 3000"
    )

    if st.button("🗑️ Clear chat"):
        st.session_state.chat = []
        st.rerun()


# ============================================================
# Upload CSV
# ============================================================

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    sig = file_signature(uploaded_file)
    if sig != st.session_state.file_sig:
        try:
            st.session_state.df = clean_dataframe(load_csv(uploaded_file))
            st.session_state.file_sig = sig
            st.session_state.agent = None  # force agent rebuild
            st.session_state.chat = []  # new dataset, new chat
            st.success("✅ Dataset loaded successfully!")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

if st.session_state.df is None:
    st.info("👆 Upload a CSV file to begin.")
    st.stop()

df = st.session_state.df

# Debug info (safe to delete once everything works)
st.sidebar.divider()
st.sidebar.caption(f"Streamlit {st.__version__} | pandas {pd.__version__}")
st.sidebar.caption(f"Loaded shape: {df.shape}")

if df.empty:
    st.error(
        "The CSV loaded but contains no rows. Check that the file has data "
        "below the header row."
    )
    st.stop()


# ============================================================
# Agent (cached per dataset + model)
# ============================================================

agent_key = (st.session_state.file_sig, model_name)

if st.session_state.agent is None or st.session_state.agent_key != agent_key:
    if not os.getenv("GROQ_API_KEY"):
        st.error("Set GROQ_API_KEY in your .env file, then restart the app.")
        st.stop()
    st.session_state.charts = []
    st.session_state.agent = build_agent(df, model_name, st.session_state.charts)
    st.session_state.agent_key = agent_key


def invoke_with_retry(history):
    """Run the agent; on malformed tool calls, retry with other models."""
    attempts = [model_name] + [m for m in MODEL_OPTIONS if m != model_name]
    last_err = None
    for attempt_model in attempts:
        try:
            if attempt_model == model_name:
                agent = st.session_state.agent
            else:
                agent = build_agent(df, attempt_model, st.session_state.charts)
            return agent.invoke({"messages": history}), attempt_model
        except Exception as e:
            last_err = e
            text = str(e)
            if "tool_use_failed" not in text and "Failed to parse tool call" not in text:
                raise  # a different error: don't hide it
            st.session_state.charts.clear()
    raise last_err


# ============================================================
# Tabs
# ============================================================

tab_preview, tab_profile, tab_chat = st.tabs(
    ["📄 Preview", "🔍 Data Profile", "💬 Ask Your Dataset"]
)


# ---------------- Preview ----------------
with tab_preview:
    st.subheader("Dataset Preview")

    if len(df) > 5:
        rows_to_show = st.slider(
            "Rows to preview", 5, min(100, len(df)), min(10, len(df))
        )
    else:
        rows_to_show = len(df)
    show_df(df.head(rows_to_show))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

    st.subheader("Columns")
    st.write(list(df.columns))


# ---------------- Profile ----------------
with tab_profile:
    st.subheader("Column Types & Missing Values")
    profile = pd.DataFrame(
        {
            "column": [str(c) for c in df.columns],
            "dtype": [str(t) for t in df.dtypes],
            "missing": df.isna().sum().values,
            "missing_%": (df.isna().mean().values * 100).round(2),
            "unique": [df[c].nunique() for c in df.columns],
        }
    )
    show_df(profile)

    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        st.subheader("Numeric Summary")
        show_df(
            numeric_df.describe().T.reset_index().rename(columns={"index": "column"})
        )

        if numeric_df.shape[1] > 1:
            st.subheader("Correlation Heatmap")
            fig = px.imshow(
                numeric_df.corr(),
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
            )
            show_chart(fig, key="corr_heatmap")


# ---------------- Chat ----------------
with tab_chat:
    st.subheader("Ask Your Dataset")

    # Render history
    for m_idx, msg in enumerate(st.session_state.chat):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for i, fig in enumerate(msg.get("charts", [])):
                show_chart(fig, key=f"hist_{m_idx}_{i}")

    question = st.chat_input("Ask a question about your data...")

    if question:
        st.session_state.chat.append(
            {"role": "user", "content": question, "charts": []}
        )
        with st.chat_message("user"):
            st.markdown(question)

        # Build message history (last 10 turns) for context
        history = []
        for m in st.session_state.chat[-11:-1]:
            cls = HumanMessage if m["role"] == "user" else AIMessage
            history.append(cls(content=m["content"]))
        history.append(HumanMessage(content=question))

        with st.chat_message("assistant"):
            with st.spinner("🤖 Analyzing your dataset..."):
                st.session_state.charts.clear()
                try:
                    result, used_model = invoke_with_retry(history)
                    answer = (
                        message_text(result["messages"][-1].content)
                        or "No answer returned."
                    )
                    new_charts = list(st.session_state.charts)

                    if used_model != model_name:
                        st.caption(
                            f"ℹ️ Retried with `{used_model}` after a tool-call failure."
                        )
                    st.markdown(answer)
                    for i, fig in enumerate(new_charts):
                        show_chart(fig, key=f"new_{len(st.session_state.chat)}_{i}")

                    with st.expander("🔧 Tools used"):
                        used = [
                            f"**{m.name}**"
                            for m in result["messages"]
                            if getattr(m, "type", "") == "tool"
                        ]
                        st.markdown(", ".join(used) if used else "No tools used.")

                    st.session_state.chat.append(
                        {"role": "assistant", "content": answer, "charts": new_charts}
                    )

                except Exception as e:
                    err = f"⚠️ Agent error: {e}"
                    st.error(err)
                    st.caption(
                        "Tip: if this is a tool-call error, pick a different "
                        "model in the sidebar and ask again."
                    )
                    st.session_state.chat.append(
                        {"role": "assistant", "content": err, "charts": []}
                    )
