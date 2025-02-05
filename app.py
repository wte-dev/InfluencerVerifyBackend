from flask import Flask, request, jsonify
import google.generativeai as genai
import os
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import re
from flask_cors import CORS

# Load API keys from .env file
load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Perplexity API client
client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")

# Initialize Gemini API client
genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)
CORS(app)

def remove_duplicate_claims(claims):
    """
    Function to remove claims that have more than 85% similarity.
    Uses Cosine Similarity with TF-IDF Vectorization to compare claims.
    """
    # Step 1: Vectorize claims using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(claims)

    # Step 2: Calculate Cosine Similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Step 3: Check for duplicates
    unique_claims = []
    for i in range(len(claims)):
        is_duplicate = False
        for j in range(i):
            if cosine_sim[i, j] > 0.85:  # If similarity is greater than 85%
                is_duplicate = True
                break
        if not is_duplicate:
            unique_claims.append(claims[i])

    return unique_claims

def extract_json_from_response(response_text):
    """Extract JSON content from a string using regex"""
    match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)  # Convert string to JSON
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format"}
    return {"error": "No JSON found in response"}

def verify_claim_with_research(unique_claims):
    """Search for relevant research papers on Perplexity API based on a given claim."""
    messages = [
        {"role": "system", "content": "Find scientific papers and sources that support or provide information related to the following claim."},
        {"role": "user", "content": f"Find scientific research papers supporting or refuting the claim: '{unique_claims}'. List only the claims and the scientific research papers and sources for each claim and nothing else."}
    ]

    try:
        # Call Perplexity API to search for relevant research papers
        response = client.chat.completions.create(
            model="llama-3.1-sonar-small-128k-online",  # Perplexity Model
            messages=messages,
        )
        
        # Extract relevant research sources or references from the response
        research_sources = response.choices[0].message.content if response.choices else "No relevant research found."
        
        # Use Gemini API to analyze research and assign trust score
        system_instruction = (
            """You are a fact-checking AI assistant. Your task is to analyze health-related claims and verify their accuracy using scientific research.

                Guidelines:
                1. You MUST return output **only in JSON format**—no explanations, extra text, or formatting beyond JSON.
                2. Analyze each health claim and classify it under one of these categories:
                ✅ "Verified" (Supported by research)
                ⚠️ "Questionable" (Mixed or unclear research)
                ❌ "Debunked" (Contradicts scientific research)
                3. Assign a trust score (0-100%) based on confidence.
                4. Provide a reason summarizing the supporting or contradicting research.

                The response **MUST** strictly follow this JSON format:
                {
                    "influencer": "<INSERT_INFLUENCER_NAME>",
                    "verified_health_claims": [
                        {
                            "claim": "<CLAIM_1>",
                            "verification_status": {
                                "status": "<Verified | Questionable | Debunked>",
                                "trust_score": "<0-100>%",
                                "reason": "<Brief explanation based on research>"
                            }
                        },
                        {
                            "claim": "<CLAIM_2>",
                            "verification_status": {
                                "status": "<Verified | Questionable | Debunked>",
                                "trust_score": "<0-100>%",
                                "reason": "<Brief explanation based on research>"
                            }
                        }
                    ]
                }

                Example:
                User Input: "Analyze health claims from Dr. Andrew Huberman:  
                1. Cold exposure can enhance dopamine levels.  
                2. Drinking alkaline water prevents cancer."

                Expected JSON Output:
                {
                    "influencer": "Dr. Andrew Huberman",
                    "verified_health_claims": [
                        {
                            "claim": "Cold exposure can enhance dopamine levels.",
                            "verification_status": {
                                "status": "Verified",
                                "trust_score": "92%",
                                "reason": "Multiple studies confirm dopamine increase after cold exposure."
                            }
                        },
                        {
                            "claim": "Drinking alkaline water prevents cancer.",
                            "verification_status": {
                                "status": "Debunked",
                                "trust_score": "20%",
                                "reason": "No scientific evidence supports this claim; research contradicts it."
                            }
                        }
                    ]
                }

                IMPORTANT:  
                - **Do not include any additional text, explanations, or formatting outside JSON.**  
                - **Ensure valid JSON output every time.**  
                - **If a claim cannot be verified, return it as 'Questionable' with an appropriate trust score.**"""
        )

        prompt = f"Analyze the following research results and determine verification status for the claim: '{unique_claims}'.\n\nResearch Results:\n{research_sources}"

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system_instruction
        )

        response_gemini = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=500,
                temperature=0.1,
            )
        )

        verification_result = response_gemini.text.strip()
        
        # Extract JSON from verification_result
        parsed_json = extract_json_from_response(verification_result)

        return parsed_json

    except Exception as e:
        return {"error": str(e)}

@app.route('/fetch-health-claims', methods=['POST'])
def fetch_health_claims():
    """Fetch recent health-related claims for an influencer and extract them using Gemini API."""
    data = request.json
    influencer_name = data.get("influencer", "").strip()

    if not influencer_name:
        return jsonify({"error": "Influencer name is required"}), 400

    # Step 1: Query Perplexity API with Llama-3.1 Sonar Model
    messages = [
        {"role": "system", "content": "Find recent health-related claims or statements made by this influencer."},
        {"role": "user", "content": f"Fetch and list down 5 most recent health-related claims made by {influencer_name}. Strictly only give the list of claims and nothing else"}
    ]

    try:
        # Call Perplexity API for content related to influencer's claims
        response = client.chat.completions.create(
            model="llama-3.1-sonar-small-128k-online",  # Perplexity Model
            messages=messages,
        )

        claims_content = response.choices[0].message.content if response.choices else "No claims found."
        
        # Step 2: Send the content to Gemini API for structured health claims extraction
        system_instruction = (
            "You are a health claims extraction assistant. Your task is to identify and categorize health-related claims from text."
        )
        prompt = f"Extract health-related claims from the following content: {claims_content}. Strictly only give the list of claims and nothing else."

        # Generate structured content using Gemini API
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",  # Gemini model
            system_instruction=system_instruction
        )
        
        response_gemini = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=1000,
                temperature=0.1,
            )
        )

        extracted_claims = response_gemini.text.strip().split('\n')  # Assuming claims are line-separated
        if extracted_claims:
            # Step 3: Remove duplicate claims based on similarity
            unique_claims = remove_duplicate_claims(extracted_claims)
        else:
            unique_claims = []

        # Step 4: Cross-reference claims with Perplexity API to search for relevant research
        verified_claims = []
        for claim in unique_claims:
            verification_result = verify_claim_with_research(claim)  # Fix: Pass single claim instead of full list
            print("Verification Result:", verification_result)  # Print verification result
            verified_claims.append(verification_result)  # Append each result

        return jsonify({
            "influencer": influencer_name,
            "extracted_health_claims": unique_claims,
            "verified_health_claims": verified_claims,  # Fix: Return the full list of verified claims
            "fetched_claims": claims_content  # Fix: Proper key format
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
