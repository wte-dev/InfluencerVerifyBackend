from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enables CORS for all routes

@app.route('/analyze', methods=['POST'])
def analyze_health_claims():
    data = request.json
    influencer = data.get("influencer", "Unknown")
    
    response_data = {
        "extracted_health_claims": [
            "* Oral health is critical for dental and microbiome health, general physical well-being, and mental health. Proper oral care can offset metabolic, cardiac, and brain diseases, including dementia.",
            "* Fluoride is essential for oral health, but excessive use can be detrimental.",
            "* Molecules in some sunscreens can be found in neurons 10 years after application.",
            "* Regular cold and heat exposure (e.g., sauna and ice bath) is beneficial.",
            "* Some dietary supplements lack scientific evidence supporting their effectiveness."
        ],
        "fetched_claims": "1. **Oral Health Importance**: Oral health is critical for dental and microbiome health, general physical well-being, and mental health. Proper oral care can offset metabolic, cardiac, and brain diseases, including dementia[2].\n2. **Fluoride Use**: Fluoride is essential for oral health, but excessive use can be detrimental. Huberman discusses the importance of fluoride in water and toothpaste, while also mentioning zero-fluoride toothpastes[2].\n3. **Sunscreen Skepticism**: Huberman has expressed skepticism about sunscreen, claiming that molecules in some sunscreens can be found in neurons 10 years after application, although he denies being an \"anti-sunscreen truther\"[1].\n4. **Cold and Heat Exposure**: Huberman advocates for regular cold and heat exposure, recommending 20 minutes in a sauna and 5 minutes in an ice bath or cold shower, repeated three to five times[3].\n5. **Dietary Supplements Criticism**: Huberman has been criticized for promoting poorly regulated dietary supplements, which often lack scientific evidence supporting their effectiveness[1].",
        "influencer": influencer,
        "verified_health_claims": [
            {
                "influencer": None,
                "verified_health_claims": [
                    {
                        "claim": "* Oral health is critical for dental and microbiome health, general physical well-being, and mental health. Proper oral care can offset metabolic, cardiac, and brain diseases, including dementia.",
                        "verification_status": {
                            "reason": "The provided research papers support the claim that oral health is significantly linked to overall physical and mental well-being. Studies indicate correlations between poor oral health and increased risks of metabolic, cardiac, and neurodegenerative diseases, including dementia. While not all studies definitively prove causation, the associations are strong enough to support the claim with a high degree of confidence.",
                            "status": "Verified",
                            "trust_score": "85%"
                        }
                    }
                ]
            },
            {
                "influencer": "Research Results on Fluoride",
                "verified_health_claims": [
                    {
                        "claim": "Fluoride is essential for oral health, but excessive use can be detrimental.",
                        "verification_status": {
                            "reason": "Numerous studies support fluoride's crucial role in preventing tooth decay and strengthening enamel. However, extensive research also demonstrates that excessive fluoride intake can lead to dental fluorosis, systemic toxicity, skeletal fluorosis, and potential neurodevelopmental issues.",
                            "status": "Verified",
                            "trust_score": "95%"
                        }
                    }
                ]
            },
            {
                "influencer": None,
                "verified_health_claims": [
                    {
                        "claim": "Molecules in some sunscreens can be found in neurons 10 years after application.",
                        "verification_status": {
                            "reason": "No research directly supports or refutes the claim of sunscreen molecules in neurons 10 years post-application. Existing research focuses on systemic absorption and persistence, but not long-term neuronal accumulation.",
                            "status": "Questionable",
                            "trust_score": "20%"
                        }
                    }
                ]
            },
            {
                "influencer": "Multiple Research Papers",
                "verified_health_claims": [
                    {
                        "claim": "Regular cold and heat exposure (e.g., sauna and ice bath) is beneficial.",
                        "verification_status": {
                            "reason": "Multiple studies show benefits such as reduced social anxiety, improved immune system function, cardiovascular benefits, neurological benefits, metabolic benefits, improved brain health, reduced risk of dementia, temporary relief from DOMS, enhanced recovery, improved mood, decreased stress, potential slowing of neurodegenerative diseases, pain management, reduced risk of cardiovascular disease, and activation of heat shock proteins. However, more research is needed to fully understand the long-term effects and optimal protocols for cold and heat exposure.",
                            "status": "Verified",
                            "trust_score": "85%"
                        }
                    }
                ]
            },
            {
                "influencer": None,
                "verified_health_claims": [
                    {
                        "claim": "Some dietary supplements lack scientific evidence supporting their effectiveness.",
                        "verification_status": {
                            "reason": "Multiple research papers and reviews show a lack of strong evidence supporting the effectiveness of many dietary supplements for weight loss and prevention of chronic diseases in healthy individuals. Some studies even suggest potential harm from high-dose supplements, and regulatory issues contribute to unsubstantiated claims.",
                            "status": "Verified",
                            "trust_score": "95%"
                        }
                    }
                ]
            }
        ]
    }
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
