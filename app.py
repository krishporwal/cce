import os
import json
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS for cross-origin requests
from PIL import Image
import io

# --- Initialization ---
app = Flask(__name__)
# Enable CORS to allow your Netlify frontend to call this backend
CORS(app)

# --- Configure the Gemini API ---
# It's crucial to set your API key as an environment variable for security
# In your terminal, run: export GEMINI_API_KEY='YOUR_API_KEY_HERE'
try:
    # This line is corrected to properly load the key from the environment.
    # NEVER paste your actual key directly into the code.
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
except (ValueError, KeyError) as e:
    print(f"CRITICAL ERROR: {e}")
    # In a real app, you might exit or handle this more gracefully
    # For now, we'll let it run but endpoints will fail.

# --- AI Model Initialization ---
# Initialize the models you'll be using
# UPDATED: Using gemini-1.5-flash-latest for both vision and text.
# This is a newer, faster, and more reliable multimodal model.
# This change fixes the "404 models/gemini-pro-vision is not found" error.
model = genai.GenerativeModel('gemini-1.5-flash-latest')


# --- Prompts for the AI ---
# A detailed prompt for image analysis to ensure structured JSON output
image_analysis_prompt = """
You are an expert plant pathologist. Analyze the provided image of a plant leaf.
Identify the plant and any disease you detect.
Provide your response ONLY in a valid JSON format with the following keys:
- "plant": The common name of the plant (e.g., "Grapevine").
- "disease": The common name of the disease (e.g., "Black Rot"). If healthy, state "Healthy".
- "description": A brief, one-paragraph description of the disease and its symptoms visible in the image.
- "confidence": Your confidence level as "High", "Medium", or "Low".

Do not include any other text or explanations outside of the JSON object.
"""

# A prompt for generating a treatment plan
treatment_plan_prompt_template = """
You are an agricultural advisor. A farmer has identified "{disease_name}" on their plant.
Generate a concise, actionable treatment plan. Structure the plan with the following sections:
1.  **Immediate Actions:** What to do right now (e.g., pruning, isolating plants).
2.  **Organic Solutions:** Safe, organic treatment options.
3.  **Chemical Solutions:** Common and effective fungicides or treatments to recommend.
4.  **Prevention Tips:** How to prevent this issue in the future.

Keep the language clear and easy for a non-expert to understand.
"""

# --- API Endpoints ---

@app.route('/analyze-image', methods=['POST'])
def analyze_image_endpoint():
    """Endpoint to receive an image and return a Gemini Vision analysis."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        img = Image.open(file.stream)
        
        # Call the Gemini API using the updated model
        response = model.generate_content([image_analysis_prompt, img])
        
        # Clean the response to ensure it's valid JSON
        json_text = response.text.strip().replace('```json', '').replace('```', '')
        result = json.loads(json_text)
        
        return jsonify(result)

    except Exception as e:
        print(f"Error in /analyze-image: {e}")
        return jsonify({"error": f"An error occurred during analysis: {str(e)}"}), 500

@app.route('/get-treatment-plan', methods=['POST'])
def get_treatment_plan_endpoint():
    """Endpoint to receive a disease name and return a generated treatment plan."""
    data = request.get_json()
    if not data or 'disease_name' not in data:
        return jsonify({"error": "Missing 'disease_name' in request body"}), 400

    disease_name = data['disease_name']
    
    try:
        # Format the prompt with the specific disease name
        prompt = treatment_plan_prompt_template.format(disease_name=disease_name)
        
        # Call the Gemini API using the updated model
        response = model.generate_content(prompt)
        
        return jsonify({"plan": response.text})

    except Exception as e:
        print(f"Error in /get-treatment-plan: {e}")
        return jsonify({"error": f"An error occurred while generating the plan: {str(e)}"}), 500

# --- Main execution ---
if __name__ == '__main__':
    # The server will run on http://1.0.0.1:5000 by default
    app.run(debug=True, port=5000)
