import os
import google.generativeai as genai

def test_api_key(api_key):
    try:
        # Configure the API key
        genai.configure(api_key=api_key)
        
        # Initialize the Gemini model
        gemini_model = genai.GenerativeModel("gemini-1.5-pro-001")
        
        # Send a simple test prompt
        test_prompt = "Hello, world!"
        response = gemini_model.generate_content(test_prompt)
        
        # If the response is returned successfully, print it
        print("API Key is functional. Response:")
        print(response.text)
    except Exception as e:
        # Print an error if something goes wrong
        print("Error: API key might be invalid or there was a connection issue.")
        print(e)

if __name__ == "__main__":
    # You can set the API key via the environment variable or directly here
    api_key = "AIzaSyBoGabhtqv5-DAPGV37jw6RAsvwadu6AkA"
    test_api_key(api_key)
