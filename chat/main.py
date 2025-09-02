import functions_framework
import json
from flask import jsonify, request
import requests
from flask_cors import CORS

# gcloud run deploy create-realtime-chat-session `
#   --source=. `
#   --region=europe-west2 `
#   --allow-unauthenticated
# ensure you are in the right directory with main.py and requirements.txt


@functions_framework.http
def create_session(request):
    """HTTP Cloud Function for creating OpenAI WebRTC sessions."""
    # Enable CORS
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "120",
        }
        return ("", 204, headers)

    headers = {"Access-Control-Allow-Origin": "*"}

    try:
        request_json = request.get_json()
        if not request_json or "api_key" not in request_json:
            return (jsonify({"error": "Missing API key"}), 400, headers)

        api_key = request_json["api_key"]
        model = request_json.get("model", "gpt-realtime")
        voice = request_json.get("voice", "verse")
        instructions = request_json.get(
            "instructions",
            "Be a friendly chatbot, speak in British English. Indicate you have been given default instructions when you first speak.",
        )

        # Log what we're sending to OpenAI
        print("Sending to OpenAI:", json.dumps(request_json, indent=2))

        # Forward request to OpenAI with beta header
        response = requests.post(
            "https://api.openai.com/v1/realtime/sessions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "OpenAI-Beta": "realtime=v1",
                "OpenAI-Organization": request_json.get(
                    "organization_id", ""
                ),  # Optional org ID
            },
            json={"model": model, "voice": voice, "instructions": instructions},
        )

        if response.status_code != 200:
            print(f"OpenAI Error: {response.status_code}")
            print(f"Response text: {response.text}")
            return (jsonify({"error": response.text}), response.status_code, headers)

        result = response.json()
        print(f"OpenAI Success Response: {json.dumps(result, indent=2)}")
        return (result, 200, headers)

    except Exception as e:
        print(f"Error in create_session: {str(e)}")
        return (jsonify({"error": str(e)}), 500, headers)


@functions_framework.http
def hello_get(request):
    return "Hello World!"
