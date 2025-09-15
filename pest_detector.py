# pest_detector.py

import os
import sys
import base64
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# --- 1. CONFIGURATION ---
# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("❌ GOOGLE_API_KEY not found. Please set it in your .env file.")
    sys.exit(1)

# --- 2. HELPER FUNCTION ---
def image_to_base64(image_path: str) -> str:
    """Converts an image file to a base64 encoded string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"❌ Error: Image file not found at '{image_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An error occurred while processing the image: {e}")
        sys.exit(1)

# --- 3. CORE ANALYSIS FUNCTION ---
def analyze_image_for_pests(image_path: str):
    """
    Analyzes an image for pests and diseases using the Gemini Vision model.
    """
    print(f"🌿 Analyzing image: {image_path}...")

    # Initialize the Gemini Vision model through LangChain
    # NEW LINE
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
    
    # Convert the image to base64
    image_b64 = image_to_base64(image_path)

    # Craft the detailed prompt for the model
    prompt_text = """
    You are an expert agricultural entomologist and plant pathologist.
    Your task is to analyze the provided image of a plant to identify any pests or diseases.

    Please provide your analysis in a structured format with the following sections:
    1.  **Identification**: Name the primary pest or disease identified. If none, state "No Pests or Diseases Detected".
    2.  **Confidence Level**: (e.g., High, Medium, Low).
    3.  **Description**: Briefly describe the visual evidence (e.g., "Small, green, pear-shaped insects clustered on the underside of leaves").
    4.  **Potential Damage**: Explain the harm this pest or disease can cause to the plant (e.g., "Sucks sap, causing yellowing, stunted growth, and sooty mold").
    5.  **Recommended Actions**: Suggest actionable, eco-friendly, and chemical treatment options.
        - **Organic/Eco-Friendly**:
        - **Chemical**:
    6.  **Disclaimer**: Add a disclaimer that this is an AI-generated analysis and a professional should be consulted for confirmation.
    """

    # Create the message payload for the model
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt_text,
            },
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image_b64}",
            },
        ]
    )

    # Invoke the model and get the response
    print("🤖 Contacting Gemini Vision API...")
    try:
        response = llm.invoke([message])
        print("✅ Analysis Complete! Here is the report:\n")
        print(response.content)
    except Exception as e:
        print(f"❌ An error occurred during the API call: {e}")

# --- 4. SCRIPT EXECUTION ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pest_detector.py <path_to_image>")
        sys.exit(1)

    image_file_path = sys.argv[1]
    analyze_image_for_pests(image_file_path)
    