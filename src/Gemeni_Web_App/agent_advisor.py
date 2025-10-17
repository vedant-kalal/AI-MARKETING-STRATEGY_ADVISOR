import os
import google.generativeai as genai

def marketing_advice(prob, contact, previous, education, age, marital, loan, model="gemini-pro-latest"):
    
    # Uses Gemini to generate marketing advice based on predicted probability.
    # Directly uses probability (as %), contact, previous, education, age, marital, and loan.

    # takes probablity with saftey ,if prob is not a number set to None
    try:
        prob_num = int(prob) 
    except (TypeError, ValueError):
        prob_num = None

    if prob_num is None:
        tier = "UNKNOWN"
    elif prob_num >= 70:
        tier = "HIGH"
    elif prob_num >= 40:
        tier = "MEDIUM"
    else:
        tier = "LOW"


    prompt = f"""
You are an **AI Marketing Strategist** for a bank. Your task is to analyze the client's details and predict how to approach them for a potential marketing campaign, Be concise, actionable, and motivating.


### --- Client Information  (summarize briefly first) ---:
With the given details, start by presenting a short, friendly summary of the client in bullet points (1 line each):
- Age: {age}
- Education: {education}
- Marital Status: {marital}
- Has Loan: {loan}
- Contact Type: {contact}
- Previous Contacts: {previous}
- Subscription Probability: {prob}%

Each point should be clear, concise, and human-like — for example:  
“Age: 35 — a young working professional with growing financial aspirations.”  

---

### --- Main Task for You (Gemini)--- :
Now act as an **AI Marketing Strategist** and create a short, effective strategy based on the client's subscription probability.

1️ **If this is a HIGH Probability Lead (≥70%)**:  
   - Write a **positive, encouraging marketing summary** that appreciates the client’s potential interest.  
   - Suggest the **best communication channel** based on their contact type and likely response time.  
   - Provide **3–5 short actionable tips**, each around 100 words, focused on converting this lead quickly.  
   - Tips should feel motivating, creative, and practical — like a senior strategist advising a marketing team.  
   - End with a powerful **motivational closing lines** (at least of 3 lines).

2️ **If this is a MEDIUM Probability Lead (40–69%)**:  
   - Write a **balanced and nurturing strategy** that aims to build trust and interest gradually.  
   - Provide **3–5 short but deep bullet points (100 words each)** on how to maintain engagement.  
   - Suggest one **personalized action** (like offering financial insights, loyalty perks, etc.).  
   - Recommend an **ideal follow-up time and channel** (e.g., “Call after 3 days” or “Send personalized email within a week”).  
   - End with an encouraging but realistic closing line.

3️ **If this is a LOW Probability Lead (<40%)**:  
   - Briefly explain **why this client might not be an ideal target right now** (based on their features like education, contact type, or previous interactions).  
   - Provide **3–5 thoughtful bullet points (100 words each)** that focus on long-term relationship-building.  
   - Suggest an **alternative product or non-marketing action**, such as offering account upgrades, community support, or relationship improvement.  
   - End with a **hopeful, professional closing line**, focusing on the brand’s care for the client’s long-term satisfaction.

---

### --- Output Format (Strictly Follow This) ---:
**Headline:**  
A clear, catchy title describing the lead type (e.g., “ ↑ High Probability Lead — Ready to Convert”, “ ~ Medium Lead — Nurture With Care”, “ ↓ Low Lead — Strengthen Relationship”).

**Client Overview:**  
5–7 concise bullet points summarizing the profile (each under 25 words).

**Strategy Plan:**  
3–5 rich, detailed bullet points (~100 words each) with practical marketing actions.

**Closing Line:**  
One short, motivating, and human line to wrap up.

---

✰ Tone Guidelines:
- Be **professional, personalized, and confident**.  
- Avoid robotic or generic advice.  
- Make each section feel like it was written by a **real marketing strategist**, not AI.  
- Adapt the writing style to the lead’s probability level — positive and urgent for high, nurturing for medium, and understanding for low.

Now, write your complete marketing recommendation below.
""".strip()

    # pretty display for probability in messages
    if prob_num is not None:
        display_prob = f"{prob_num}%"   
    else:
        display_prob = "N/A"

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        # if Gemini API key not found
        return f"{tier} probability ({display_prob}) lead via {contact}. Suggest personalized approach."

    try:
        # Load Gemini API key and generate content
        genai.configure(api_key=key)
        response = genai.GenerativeModel(model).generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini error: {e}"
