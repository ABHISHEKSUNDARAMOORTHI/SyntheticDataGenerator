import streamlit as st
import pandas as pd
import json
from faker import Faker
import random
from io import StringIO
import re # Import the regex module
# Import the ask_gemini function from the new utility file
from gemini_utils import ask_gemini

# Initialize Faker
fake = Faker()

# Define initial_schema_guidance_json at the top-level scope
initial_schema_guidance_json = """
{
    "id": {"type": "int", "params": {"min": 1, "max": 10000}},
    "user_name": {"type": "name"},
    "email_address": {"type": "email"},
    "account_status": {"type": "categorical", "params": {"choices": ["active", "inactive", "suspended"]}},
    "registration_date": {"type": "date", "params": {"start_date": "-5y", "end_date": "today"}},
    "transaction_amount_usd": {"type": "float", "params": {"min": 10.0, "max": 5000.0, "precision": 2}},
    "is_premium_user": {"type": "boolean"},
    "api_response_code": {"type": "status_code"},
    "log_message": {"type": "sentence"}
}
"""

def generate_synthetic_data(schema: dict, num_rows: int = 10) -> pd.DataFrame:
    """Generates synthetic data based on schema with robust error handling"""
    data = {}

    for col_name, col_info in schema.items():
        if not isinstance(col_info, dict):
            st.error(f"Invalid schema for '{col_name}' - expected a dictionary for column definition.")
            continue

        col_type = col_info.get('type', 'string')
        col_params = col_info.get('params', {})

        try:
            # Standard types for Faker
            if col_type == 'name':
                data[col_name] = [fake.name() for _ in range(num_rows)]
            elif col_type == 'email':
                data[col_name] = [fake.email() for _ in range(num_rows)]
            elif col_type == 'address':
                data[col_name] = [fake.address() for _ in range(num_rows)]
            elif col_type == 'phone_number':
                data[col_name] = [fake.phone_number() for _ in range(num_rows)]
            elif col_type == 'date':
                start = col_params.get('start_date', '-10y')
                end = col_params.get('end_date', 'today')
                data[col_name] = [fake.date_between(start_date=start, end_date=end) for _ in range(num_rows)]
            elif col_type == 'iso_date':
                data[col_name] = [fake.date_time_between(start_date='-2y', end_date='now').isoformat() for _ in range(num_rows)]
            elif col_type in ['int', 'integer']:
                data[col_name] = [random.randint(
                    col_params.get('min', 0),
                    col_params.get('max', 100)
                ) for _ in range(num_rows)]
            elif col_type == 'float':
                data[col_name] = [round(random.uniform(
                    col_params.get('min', 0.0),
                    col_params.get('max', 100.0)
                ), col_params.get('precision', 2)) for _ in range(num_rows)]
            elif col_type == 'sentence':
                data[col_name] = [fake.sentence() for _ in range(num_rows)]
            elif col_type == 'string':
                data[col_name] = [fake.word() for _ in range(num_rows)]
            elif col_type == 'boolean':
                data[col_name] = [random.choice([True, False]) for _ in range(num_rows)]
            elif col_type == 'status_code':
                choices = col_params.get('choices', [200, 201, 400, 401, 403, 404, 500])
                if isinstance(choices, list) and all(isinstance(c, int) for c in choices):
                    data[col_name] = [random.choice(choices) for _ in range(num_rows)]
                elif 'min_code' in col_params and 'max_code' in col_params: # Handle min_code/max_code from user's example
                    min_c = col_params['min_code']
                    max_c = col_params['max_code']
                    data[col_name] = [random.randint(min_c, max_c) for _ in range(num_rows)]
                else:
                    st.warning(f"No valid choices or range provided for status_code column '{col_name}'. Filling with default status codes.")
                    data[col_name] = [random.choice([200, 201, 400, 401, 403, 404, 500]) for _ in range(num_rows)]
            elif col_type == 'categorical':
                choices = col_params.get('choices', ['option1', 'option2', 'option3'])
                if not choices:
                    st.warning(f"No choices provided for categorical column '{col_name}'. Filling with empty strings.")
                    data[col_name] = [''] * num_rows
                else:
                    data[col_name] = [random.choice(choices) for _ in range(num_rows)]
            # Handle 'uuid' type from user's example
            elif col_type == 'uuid':
                import uuid
                data[col_name] = [str(uuid.uuid4()) for _ in range(num_rows)]
            # Handle 'name' with min/max_length from user's example
            elif col_type == 'name' and 'min_length' in col_params: # Assuming Faker handles length implicitly
                data[col_name] = [fake.name() for _ in range(num_rows)] # Faker doesn't have direct min/max length for names
            # Handle 'email' with domain/username length from user's example
            elif col_type == 'email' and 'domain' in col_params: # Faker doesn't have direct username length control
                data[col_name] = [fake.email(domains=[col_params['domain']]) if col_params.get('domain') else fake.email() for _ in range(num_rows)]
            # Handle 'sentence' with min/max_words from user's example
            elif col_type == 'sentence' and 'min_words' in col_params:
                min_w = col_params.get('min_words', 5)
                max_w = col_params.get('max_words', 20)
                data[col_name] = [fake.sentence(nb_words=random.randint(min_w, max_w)) for _ in range(num_rows)]

            # Fallback for unsupported types
            else:
                st.warning(f"Unsupported type '{col_type}' for column '{col_name}'. Falling back to generic string.")
                data[col_name] = [str(fake.word()) for _ in range(num_rows)]

        except Exception as e:
            st.error(f"Error generating data for column '{col_name}': {str(e)}")
            with st.expander("Show error details"): # Wrap error details
                st.exception(e)
            data[col_name] = [None] * num_rows # Fill with None on error

    # Ensure all lists in `data` have `num_rows` elements before creating DataFrame
    for col_name in schema.keys():
        if col_name not in data or len(data[col_name]) != num_rows:
            data[col_name] = [None] * num_rows

    return pd.DataFrame(data)

# Streamlit UI Configuration
st.set_page_config(
    page_title="Synthetic Data Generator",
    page_icon="🧊",
    layout="wide", # Keeping wide to allow side-by-side tables
    initial_sidebar_state="collapsed" # No primary use for sidebar anymore
)

# Custom CSS for report-like styling and layout
st.markdown("""
<style>
/* General app background and text colors for dark theme */
.stApp {
    background-color: #0E1117; /* Streamlit's default dark theme background */
    color: #FAFAFA; /* Streamlit's default dark theme text */
}

/* Adjust padding for the main content area for a more spacious feel */
.stApp > header {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 1rem;
    background-color: #1a1a1a; /* Darker header background */
    border-bottom: 1px solid #26272b; /* Subtle border */
    position: sticky; /* Make header sticky */
    top: 0;
    z-index: 999;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}
.stApp > div:first-child > div:nth-child(2) { /* Targeting the main content block */
    padding-left: 2rem;
    padding-right: 2rem;
    padding-bottom: 2rem;
    /* Allow horizontal scrolling for the main content block if needed */
    overflow-x: auto;
}

/* Headings */
.big-font {
    font-size:3em !important;
    font-weight: bold;
    color: #4CAF50; /* Green for main title */
    text-align: center; /* Center the main title */
    margin-bottom: 0.5em;
}
.medium-font {
    font-size:1.7em !important;
    font-weight: bold;
    margin-top: 2em; /* More space above sections */
    margin-bottom: 0.8em;
    color: #ADD8E6; /* Light blue for medium headings */
    border-bottom: 2px solid #26272b; /* Underline for sections */
    padding-bottom: 0.5em;
}
h4 { /* Sub-headings for insights */
    font-size: 1.3em !important;
    font-weight: bold;
    margin-top: 1em;
    margin-bottom: 0.5em;
    color: #FFD700; /* Gold for sub-headings */
}

/* Info box styling for report sections */
.stInfo {
    border-left: 5px solid #1e90ff;
    padding: 10px;
    background-color: #1a1a1a;
    border-radius: 5px;
    border: 1px solid #1e90ff;
    color: #FAFAFA;
    margin-bottom: 1em;
}

/* DataFrame styling for a clear report-like table look */
.stDataFrame table {
    border-collapse: collapse;
    width: 100% !important;
    margin-bottom: 1.5em;
    border: 1px solid #444444;
    border-radius: 5px;
    overflow: hidden;
}

.stDataFrame table th {
    background-color: #26272b;
    color: #FAFAFA;
    border: 1px solid #444444;
    padding: 10px 15px;
    text-align: left;
    font-weight: bold;
    font-size: 0.95em;
}

.stDataFrame table td {
    border: 1px solid #444444;
    padding: 8px 15px;
    vertical-align: top;
    font-size: 0.9em;
}

/* Zebra stripping for table rows */
.stDataFrame table tbody tr:nth-child(even) {
    background-color: #1e1e1e;
}
.stDataFrame table tbody tr:nth-child(odd) {
    background-color: #0E1117;
}

.stDataFrame table tbody tr:hover {
    background-color: #2b2b2b;
}

/* Ensure stDataFrame wrapper takes 100% of its parent column */
.stDataFrame {
    width: 100% !important;
    /* Apply max-height and overflow-y for preview table to avoid excessive scrolling */
    max-height: 400px; /* Limit height of preview table */
    overflow-y: auto;
}
/* For charts, allow horizontal scrolling if they overflow */
.stPlotlyChart, .stBarChart, .stLineChart {
    overflow-x: auto;
}

/* Adjust Streamlit Tabs styling */
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.2rem;
    font-weight: bold;
    color: #ADD8E6;
}
.stTabs [data-baseweb="tab-list"] button.st-emotion-cache-1215i5j [data-testid="stMarkdownContainer"] p {
    color: #4CAF50;
}

/* Adjust Streamlit Expander styling */
.streamlit-expanderHeader {
    background-color: #1a1a1a;
    border-radius: 5px;
    border: 1px solid #444444;
    padding: 10px;
    margin-bottom: 0.5em;
    color: #FAFAFA;
    font-weight: bold;
}
.streamlit-expanderContent {
    background-color: #0E1117;
    border: 1px solid #444444;
    border-top: none;
    border-radius: 0 0 5px 5px;
    padding: 15px;
    margin-bottom: 1em;
}

/* Specific styling for the main content block to give it a "card" feel if desired */
.main .block-container {
    background-color: #1a1a1a; /* Darker background for main content area */
    border-radius: 10px;
    padding: 2rem;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3); /* Subtle shadow for depth */
    margin-top: 1rem; /* Space below header */
}

/* Mobile responsiveness */
@media screen and (max-width: 768px) {
  .stApp .element-container {
    flex-direction: column !important; /* Stack columns vertically on small screens */
  }
  .stDataFrame, .stTabs, .stPlotlyChart, .stBarChart, .stLineChart {
    width: 100% !important; /* Ensure they take full width */
  }
  /* Remove padding to maximize space on small screens */
  .stApp > header, .stApp > div:first-child > div:nth-child(2) {
    padding-left: 1rem;
    padding-right: 1rem;
  }
}
</style>
""", unsafe_allow_html=True)


# Session state to hold schema across reruns
if 'current_schema' not in st.session_state:
    st.session_state['current_schema'] = {}
if 'manual_schema_text_area' not in st.session_state:
    st.session_state['manual_schema_text_area'] = initial_schema_guidance_json.strip()
# Session state for generated DataFrame
if 'generated_df' not in st.session_state:
    st.session_state['generated_df'] = pd.DataFrame()

# Session state for feedback
if 'show_feedback_form' not in st.session_state:
    st.session_state['show_feedback_form'] = False
# Session state for AI schema review
if 'ai_schema_review_output' not in st.session_state:
    st.session_state['ai_schema_review_output'] = ""
# Session state for JSON Schema explanation
if 'json_schema_input' not in st.session_state:
    st.session_state['json_schema_input'] = ""
if 'json_schema_explanation_output' not in st.session_state:
    st.session_state['json_schema_explanation_output'] = ""


# --- Top Header Section (No Sidebar) ---
st.markdown("<h1 class='big-font'>🧊 Synthetic Data Generator</h1>", unsafe_allow_html=True)
st.write("Define your schema → Generate → Analyze → Download")
st.markdown("---") # Visual separator

# --- Schema Section (Full Width with Tabs) ---
st.markdown("<h2 class='medium-font'>📐 Define Schema</h2>", unsafe_allow_html=True)

tab_define, tab_summary = st.tabs(["⚙️ Input Schema", "📊 Active Schema Summary"])

with tab_define:
    schema_choice = st.radio(
        "**Choose Input Method:**",
        ("Manual JSON Input", "Upload JSON Schema File", "Infer from Sample CSV"),
        index=0,
        key="schema_input_method_radio"
    )

    new_schema_temp = {}
    schema_parse_success = False

    if schema_choice == "Manual JSON Input":
        st.subheader("✍️ Enter Schema Manually")
        st.markdown("Define columns with types like `name`, `email`, `int`, `float`, `date`, `categorical`, `boolean`, `status_code`, etc. Use `params` for ranges or choices.")
        manual_schema_input = st.text_area(
            "Enter your schema as JSON:",
            height=300,
            value=st.session_state['manual_schema_text_area'],
            key="manual_schema_input_key"
        )
        st.session_state['manual_schema_text_area'] = manual_schema_input

        # Auto-format JSON button
        if st.button("Auto-Format JSON", key="auto_format_json_btn"):
            try:
                # Attempt to remove comments before formatting
                cleaned_input_for_format = re.sub(r"//.*?\n|/\*.*?\*/", "", manual_schema_input, flags=re.DOTALL)
                formatted_json = json.dumps(json.loads(cleaned_input_for_format), indent=4)
                st.session_state['manual_schema_text_area'] = formatted_json
                st.success("JSON formatted!")
                st.rerun()
            except json.JSONDecodeError:
                st.error("Invalid JSON to format. Please ensure syntax is correct after removing comments.")
            except Exception as e:
                st.error(f"An unexpected error occurred during formatting: {e}")


        st.markdown("---") # Small separator for example
        st.markdown("#### Example Schema Structure:")
        st.json(json.loads(initial_schema_guidance_json.strip()))

        try:
            if manual_schema_input.strip():
                # Pre-process: remove single-line comments // and block comments /* */
                # This makes the parser more forgiving to user input.
                cleaned_input = re.sub(r"//.*?\n|/\*.*?\*/", "", manual_schema_input, flags=re.DOTALL)
                new_schema_temp = json.loads(cleaned_input)
                schema_parse_success = True
            else:
                new_schema_temp = {}
                schema_parse_success = False
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {str(e)}. Please ensure no comments, trailing commas, or other syntax errors.")
            with st.expander("Show error details"): # Wrap error details
                st.exception(e)
            new_schema_temp = {}
            schema_parse_success = False


    elif schema_choice == "Upload JSON Schema File":
        st.subheader("⬆️ Upload JSON Schema File")
        uploaded_file = st.file_uploader("Upload `.json` schema file", type="json", key="upload_json_file_key")
        if uploaded_file:
            try:
                # No need to strip comments for uploaded files as they should be valid JSON
                new_schema_temp = json.load(uploaded_file)
                schema_parse_success = True
                st.success("Schema loaded successfully! ✅")
            except Exception as e:
                st.error(f"Error loading schema: {str(e)}. Please ensure the file contains valid JSON.")
                with st.expander("Show error details"): # Wrap error details
                    st.exception(e)
                new_schema_temp = {}
                schema_parse_success = False

    elif schema_choice == "Infer from Sample CSV":
        st.subheader("📄 Infer Schema from Sample CSV")
        uploaded_csv = st.file_uploader("Upload `.csv` file to infer schema", type="csv", key="upload_csv_file_key")
        if uploaded_csv:
            try:
                sample_df = pd.read_csv(uploaded_csv)
                inferred_schema = {}

                for col in sample_df.columns:
                    if pd.api.types.is_numeric_dtype(sample_df[col]):
                        if pd.api.types.is_integer_dtype(sample_df[col]):
                            inferred_schema[col] = {
                                "type": "int",
                                "params": {
                                    "min": int(sample_df[col].min()),
                                    "max": int(sample_df[col].max())
                                }
                            }
                        else:
                            inferred_schema[col] = {
                                "type": "float",
                                "params": {
                                    "min": float(sample_df[col].min()),
                                    "max": float(sample_df[col].max())
                                }
                            }
                    elif pd.api.types.is_bool_dtype(sample_df[col]):
                         inferred_schema[col] = {"type": "boolean"}
                    elif sample_df[col].nunique() < 0.1 * len(sample_df) and sample_df[col].nunique() < 50:
                        inferred_schema[col] = {
                            "type": "categorical",
                            "params": {
                                "choices": sample_df[col].dropna().unique().tolist()
                            }
                        }
                    else:
                        inferred_schema[col] = {"type": "string"}

                st.markdown("#### Inferred Schema Preview:")
                st.json(inferred_schema)
                new_schema_temp = inferred_schema
                schema_parse_success = True
                st.success("Schema inferred! You can copy and refine this in 'Manual JSON Input'.")
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")
                with st.expander("Show error details"): # Wrap error details
                    st.exception(e)
                new_schema_temp = {}
                schema_parse_success = False

    st.markdown("---")
    # Columns for Apply Schema button and AI Help button
    apply_col, ai_help_col = st.columns([0.4, 0.6])

    with apply_col:
        if st.button("✅ Apply Schema to Dashboard", type="primary", use_container_width=True, key="apply_schema_button"):
            if schema_parse_success and new_schema_temp:
                st.session_state['current_schema'] = new_schema_temp
                st.session_state['ai_schema_review_output'] = "" # Clear previous AI review on new schema apply
                st.rerun() # Trigger rerun only when explicitly applied
            elif not new_schema_temp and schema_choice != "Infer from Sample CSV":
                st.warning("No valid schema to apply. Please input or select a valid schema.")
            else:
                st.warning("Please ensure a valid schema is loaded/inferred before applying.")

    with ai_help_col:
        if st.button("🤖 Get Schema AI Help", type="secondary", use_container_width=True, key="ai_help_button"):
            schema_to_review = {}
            if schema_choice == "Manual JSON Input":
                # Use the current content of the text area for AI review, stripping comments
                try:
                    cleaned_input_for_ai = re.sub(r"//.*?\n|/\*.*?\*/", "", st.session_state['manual_schema_text_area'], flags=re.DOTALL)
                    schema_to_review = json.loads(cleaned_input_for_ai)
                except json.JSONDecodeError:
                    st.error("Please provide valid JSON in the text area (without comments) before asking for AI help.")
                    st.session_state['ai_schema_review_output'] = "Invalid JSON provided for AI review."
                    st.stop() # Stop execution to prevent further errors
            elif st.session_state['current_schema']:
                # Use the currently applied schema if available
                schema_to_review = st.session_state['current_schema']
            elif new_schema_temp and schema_parse_success:
                # Use the newly parsed schema from file/CSV if not yet applied
                schema_to_review = new_schema_temp
            else:
                st.warning("No schema available for AI review. Please define or load a schema first.")
                st.session_state['ai_schema_review_output'] = "No schema provided for AI review."
                st.stop() # Stop execution

            if schema_to_review:
                prompt = f"""
                Review the following JSON schema designed for a synthetic data generator.
                ```json
                {json.dumps(schema_to_review, indent=2)}
                ```
                Identify any potential issues or inconsistencies (e.g., conflicting data types, missing parameters for 'categorical' or 'int'/'float' types, values that might lead to errors during generation, general bad practices).
                Suggest improvements for clarity, completeness, or robustness.
                Explain typical use cases for each data type in the schema, especially how the 'params' affect generation.
                Keep the explanation concise and focused on practical advice for a data generation tool.
                If the schema looks good with no major issues, simply start your response with 'Schema looks good! ✅' followed by general tips for optimal synthetic data generation based on the provided types.
                Ensure your response is formatted using Markdown.
                """
                with st.spinner("Asking Gemini to review your schema... 🤖"):
                    gemini_output = ask_gemini(prompt)
                    st.session_state['ai_schema_review_output'] = gemini_output
            else:
                st.warning("No valid schema identified for AI review.")

    # Display AI Schema Review output if available
    if st.session_state['ai_schema_review_output']:
        with st.expander("💬 AI Schema Review", expanded=True):
            st.markdown(st.session_state['ai_schema_review_output'])
            st.caption("Powered by Gemini. Response quality may vary.")


with tab_summary:
    st.markdown("<h3 class='medium-font'>📊 Current Active Schema:</h3>", unsafe_allow_html=True)
    if st.session_state['current_schema']:
        schema = st.session_state['current_schema']
        st.json(schema)

        # Download Schema JSON button
        st.download_button(
            label="📥 Download Schema JSON",
            data=json.dumps(schema, indent=4),
            file_name="synthetic_schema.json",
            mime="application/json",
            use_container_width=True,
            key="download_schema_json_btn"
        )

        # --- Schema Summary details in this tab ---
        st.markdown("<h4 class='medium-font'>Schema Summary:</h4>", unsafe_allow_html=True)
        col_counts = {}
        for col_name, col_info in schema.items():
            col_type = col_info.get('type', 'unknown')
            col_counts[col_type] = col_counts.get(col_type, 0) + 1

        st.write(f"**Total Columns:** {len(schema)} 📦")
        for col_type, count in col_counts.items():
            st.write(f"- **{col_type.capitalize()}** columns: {count} { '📝' if col_type == 'string' else '🔢' if col_type in ['int', 'float'] else '🏷️' if col_type == 'categorical' else '📅' if col_type == 'date' else '📧' if col_type == 'email' else '📍' if col_type == 'address' else '📞' if col_type == 'phone_number' else '✅' if col_type == 'boolean' else '❓'}")
    else:
        st.info("No schema applied yet. Define one in the 'Input Schema' tab and click 'Apply Schema to Dashboard'.")

st.markdown("---") # Separator after schema section

# --- Natural Language Explanations for JSON Schemas Section ---
st.markdown("<h2 class='medium-font'>🗣️ JSON Schema Explainer</h2>", unsafe_allow_html=True)
st.write("Get clear, human-readable explanations for your JSON schemas. Enter a JSON schema below:")

json_schema_input = st.text_area(
    "Enter JSON Schema here:",
    value=st.session_state['json_schema_input'],
    height=250,
    key="json_schema_explainer_input"
)
st.session_state['json_schema_input'] = json_schema_input # Update session state

if st.button("✨ Explain JSON Schema", type="primary", use_container_width=True, key="explain_json_schema_button"):
    if json_schema_input:
        try:
            # Pre-process: remove single-line comments // and block comments /* */
            cleaned_json_input = re.sub(r"//.*?\n|/\*.*?\*/", "", json_schema_input, flags=re.DOTALL)
            parsed_schema = json.loads(cleaned_json_input) # Attempt to parse after cleaning

            prompt = f"""
            Provide a clear, human-readable explanation of the following JSON schema.
            Focus on describing what each property means, its expected data type, and any constraints or patterns.
            Explain the overall purpose or structure implied by the schema.
            ```json
            {json.dumps(parsed_schema, indent=2)}
            ```
            Format your explanation using Markdown, with clear headings and bullet points.
            """
            with st.spinner("Asking Gemini to explain the JSON schema... 🤖"):
                explanation_output = ask_gemini(prompt)
                st.session_state['json_schema_explanation_output'] = explanation_output
        except json.JSONDecodeError as e: # Catch specific JSON decoding errors
            st.error(f"Invalid JSON Schema format: {e}. Please ensure your JSON is syntactically correct (no comments, trailing commas, etc.).")
            st.session_state['json_schema_explanation_output'] = ""
        except Exception as e:
            st.error(f"An unexpected error occurred during JSON schema explanation: {e}")
            with st.expander("Show error details"):
                st.exception(e)
            st.session_state['json_schema_explanation_output'] = ""
    else:
        st.warning("Please enter a JSON schema to get an explanation.")

if st.session_state['json_schema_explanation_output']:
    with st.expander("💬 JSON Schema Explanation", expanded=True):
        st.markdown(st.session_state['json_schema_explanation_output'])
        st.caption("Powered by Gemini. Response quality may vary.")

st.markdown("---") # Separator after JSON Schema Explainer

# --- Generate Data Section ---
st.markdown("<h2 class='medium-font'>🎛️ Generate Data</h2>", unsafe_allow_html=True)

if st.session_state['current_schema']:
    num_rows = st.slider(
        "Number of rows to generate: 🔢",
        min_value=1,
        max_value=100000,
        value=100,
        step=1,
        help="Adjust the number of rows for your synthetic dataset."
    )

    gen_col, reset_col, _ = st.columns([0.2, 0.2, 0.6], gap="large")
    with gen_col:
        if st.button("✨ Generate Data", type="primary", use_container_width=True, key="generate_data_button"):
            with st.spinner(f"Generating {num_rows} rows of data..."):
                try:
                    df = generate_synthetic_data(st.session_state['current_schema'], num_rows)
                    st.session_state['generated_df'] = df
                    st.success(f"Successfully generated {num_rows} rows! 🎉")
                    st.toast("Data generated successfully! 🎉")

                    if df.isnull().sum().sum() > 0:
                        st.warning("⚠️ The generated dataset contains missing values. Check 'DataFrame Info' for details.")

                except Exception as e:
                    st.error(f"Error generating data: {str(e)}")
                    with st.expander("Show error details"): # Wrap error details
                        st.exception(e)
                    if 'generated_df' in st.session_state:
                        del st.session_state['generated_df']

    with reset_col:
        if st.button("🔄 Reset All", use_container_width=True, key="reset_all_button"):
            st.session_state['current_schema'] = {}
            st.session_state['manual_schema_text_area'] = initial_schema_guidance_json.strip()
            if 'generated_df' in st.session_state:
                del st.session_state['generated_df']
            st.session_state['ai_schema_review_output'] = "" # Clear AI review output on reset
            st.session_state['json_schema_input'] = "" # Clear JSON schema explainer input on reset
            st.session_state['json_schema_explanation_output'] = "" # Clear JSON schema explainer output on reset
            st.rerun()
else:
    st.info("⚠️ No schema is active. Please define one in the 'Define Schema' section above and click 'Apply Schema to Dashboard'.")

st.markdown("---") # Separator after generate section

# --- Output Dashboard Layout (Post-Generation) ---
st.markdown("<h2 class='medium-font'>Output Dashboard 📊</h2>", unsafe_allow_html=True)

if 'generated_df' in st.session_state and not st.session_state['generated_df'].empty:
    df = st.session_state['generated_df']

    # Use st.tabs for Preview and Quick Stats (as suggested)
    tabs_preview_stats = st.tabs(["📋 Preview", "⚡ Quick Stats"])

    with tabs_preview_stats[0]: # Preview Tab
        st.markdown(f"#### 🧪 Preview Table (First {min(10, df.shape[0])} Rows):")
        with st.container(): # Wrap preview table to control potential overflow
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    with tabs_preview_stats[1]: # Quick Stats Tab
        st.markdown("<h4 class='medium-font'>⚡️ Key Metrics:</h4>", unsafe_allow_html=True)
        # Using st.columns for metrics for a more compact look
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric(label="Total Rows", value=f"{df.shape[0]} 📏")
        with metric_col2:
            st.metric(label="Total Columns", value=f"{df.shape[1]} 📊")
        with metric_col3:
            try:
                mem_usage_mb = df.memory_usage(deep=True).sum() / (1024**2)
                st.metric(label="Memory Usage", value=f"{mem_usage_mb:.2f} MB �")
            except Exception:
                st.metric(label="Memory Usage", value="N/A 🤷")

        st.markdown("---") # Separator within tab

        # Column Types Distribution (Table + Chart) within an expander
        with st.expander("📊 Column Types Breakdown", expanded=True):
            st.markdown("<h4 class='medium-font'>🗃️ Column Types Distribution:</h4>", unsafe_allow_html=True)
            dtype_counts = df.dtypes.astype(str).value_counts().reset_index()
            dtype_counts.columns = ['Data Type', 'Count']
            st.dataframe(dtype_counts, hide_index=True, use_container_width=True)
            # Wrap bar chart in a container for potential overflow-x handling
            with st.container():
                st.markdown("<div style='overflow-x: auto;'>", unsafe_allow_html=True)
                st.bar_chart(dtype_counts.set_index('Data Type'))
                st.markdown("</div>", unsafe_allow_html=True)


        # Sample Column Insights within an expander
        with st.expander("✨ Sample Column Insights"):
            st.markdown("<h4 class='medium-font'>Unique Values from Data:</h4>", unsafe_allow_html=True)
            sample_categorical_cols = df.select_dtypes(include=['object', 'category', 'boolean']).columns
            if not sample_categorical_cols.empty:
                st.write("Here are some unique values from your data:")
                for col_idx, col in enumerate(sample_categorical_cols[:3]): # Show for up to 3 columns
                    unique_vals = df[col].value_counts().index.tolist()
                    if len(unique_vals) > 5:
                        st.write(f"**{col}:** {', '.join(map(str, unique_vals[:3]))}... (and {len(unique_vals) - 3} more)")
                    else:
                        st.write(f"**{col}:** {', '.join(map(str, unique_vals))}")
                    if col_idx < len(sample_categorical_cols[:3]) - 1:
                        st.markdown("---", unsafe_allow_html=True)
            else:
                st.info("No categorical/boolean columns to show unique value insights. 🙁")


    st.markdown("---") # Separator before detailed insights

    # --- Detailed Insights (Using Tabs) ---
    st.markdown("<h3 class='medium-font'>🔍 Detailed Generated Data Insights:</h3>", unsafe_allow_html=True)

    tabs_detailed_insights = st.tabs(["ℹ️ DataFrame Info", "📊 Numerical Stats", "🏷️ Categorical Counts"])

    with tabs_detailed_insights[0]: # DataFrame Info Tab
        st.markdown("#### DataFrame Info:")
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    with tabs_detailed_insights[1]: # Numerical Stats Tab
        numerical_cols = df.select_dtypes(include=['number']).columns
        if not numerical_cols.empty:
            st.markdown("#### Numerical Column Statistics (Line Chart):")
            # Changed to line chart for numerical stats as suggested
            st.line_chart(df[numerical_cols].describe().T)
            st.markdown("#### Numerical Column Statistics (Table):") # Still keep table as optional
            st.dataframe(df[numerical_cols].describe().T, use_container_width=True)
        else:
            st.info("No numerical columns for statistics. 🔢")

    with tabs_detailed_insights[2]: # Categorical Counts Tab
        categorical_cols = df.select_dtypes(include=['object', 'category', 'boolean']).columns
        if not categorical_cols.empty:
            st.markdown("#### Categorical/Boolean Column Value Counts (Top 10):")
            for col_idx, col in enumerate(categorical_cols):
                st.write(f"**Column: '{col}'**")
                st.dataframe(df[col].value_counts().head(10).reset_index(), use_container_width=True)
                if col_idx < len(categorical_cols) - 1:
                    st.markdown("---", unsafe_allow_html=True)
        else:
            st.info("No categorical/boolean columns for value counts. 📚")

    # --- Download Section ---
    st.markdown("---")
    st.subheader("💾 Download Generated Data")
    
    # Download buttons
    download_col1, download_col2, _ = st.columns([0.25, 0.25, 0.5])
    with download_col1:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download CSV",
            csv_data,
            f"synthetic_data_{df.shape[0]}_rows.csv",
            "text/csv",
            use_container_width=True,
            key='download-csv'
        )
    with download_col2:
        json_data = df.to_json(orient='records', indent=4).encode('utf-8')
        st.download_button(
            "⬇️ Download JSON",
            json_data,
            f"synthetic_data_{df.shape[0]}_rows.json",
            "application/json",
            use_container_width=True,
            key='download-json'
        )
    # Optional: Display file size estimate after download
    st.info(f"CSV size estimate: Approximately {len(csv_data) / 1024:.2f} KB. JSON size estimate: Approximately {len(json_data) / 1024:.2f} KB.")

else:
    st.info("Generate data to see the output dashboard here. ☝️")

st.markdown("---")

# --- Feedback Button and Form ---
if st.button("💬 Provide Feedback", key="feedback_button"):
    st.session_state['show_feedback_form'] = not st.session_state['show_feedback_form'] # Toggle visibility

if st.session_state['show_feedback_form']:
    st.subheader("Your Feedback Matters! 🌟")
    feedback_text = st.text_area("Tell us what you think:", key="feedback_text_area")
    if st.button("Submit Feedback", key="submit_feedback_button"):
        if feedback_text:
            print(f"User Feedback: {feedback_text}") # Print to console
            st.success("Thanks for your feedback! We've received it. 🎉")
            st.session_state['show_feedback_form'] = False # Hide form after submission
            st.rerun() # Rerun to clear text area
        else:
            st.warning("Please enter some feedback before submitting.")

st.caption("Synthetic Data Generator | Generate realistic test data for development and testing")