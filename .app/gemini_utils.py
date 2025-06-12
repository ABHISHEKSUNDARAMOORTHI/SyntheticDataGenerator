# gemini_utils.py

import google.generativeai as genai




# Retrieve the API key
GOOGLE_API_KEY ="YOUR_ACTUAL_GEMINI_API_KEY_HERE"

# Basic validation for API Key
if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_ACTUAL_GEMINI_API_KEY_HERE":
    # This RuntimeError will be caught by Streamlit's exception handling
    # and displayed to the user.
    raise RuntimeError(
        "GEMINI_API_KEY not found or invalid. "
        "Please replace 'YOUR_ACTUAL_GEMINI_API_KEY_HERE' "
        "with your actual Google Gemini API key in your .env file."
    )

# --- Gemini Model Initialization ---
model = None # Initialize model as None

def get_available_gemini_model():
    """
    Checks for available Gemini models that support generateContent method,
    prioritizing 'gemini-1.5-flash', then 'gemini-1.0-pro', then any other suitable model.
    """
    # Prioritize gemini-1.5-flash
    for m in genai.list_models():
        if m.name == 'models/gemini-1.5-flash' and 'generateContent' in m.supported_generation_methods:
            return genai.GenerativeModel(m.name)

    # Fallback to gemini-1.0-pro
    for m in genai.list_models():
        if m.name == 'models/gemini-1.0-pro' and 'generateContent' in m.supported_generation_methods:
            return genai.GenerativeModel(m.name)

    # Fallback to any other model supporting generateContent
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            return genai.GenerativeModel(m.name)
            
    raise Exception(
        "No Gemini model found that supports 'generateContent'. "
        "Please ensure your API key is correct and valid, or check Google AI Studio for available models."
    )

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = get_available_gemini_model()
except Exception as e:
    # This exception will be propagated to main_app.py if `ask_gemini` is called
    # and `model` is None.
    print(f"FATAL ERROR: Could not initialize Gemini model. "
          f"Ensure API key is valid and models are accessible: {e}")
    model = None # Explicitly set model to None on failure

def ask_gemini(prompt: str) -> str:
    """
    Sends a prompt to the configured Gemini model and returns the text response.
    Returns Markdown-formatted text.
    """
    if model is None:
        # If model failed to initialize, return a user-friendly error message.
        return "❌ Gemini AI service is not available. Please check your API key and model access."
    
    if not prompt.strip():
        return "Please provide a valid input for explanation."

    try:
        response = model.generate_content(prompt)
        # Check for valid content in the response
        if response and response.candidates and len(response.candidates) > 0 and \
           response.candidates[0].content and response.candidates[0].content.parts and \
           len(response.candidates[0].content.parts) > 0:
            return response.candidates[0].content.parts[0].text
        else:
            return "❌ Gemini API Error: No valid response text found. The AI might not have generated content for this query."
    except Exception as e:
        return f"❌ Gemini API Call Failed: {e}\n\n" \
               "Possible issues: incorrect API key, rate limit exceeded, or model access problems. " \
               "Please ensure your GEMINI_API_KEY is correct and you have access to the selected model(s)."