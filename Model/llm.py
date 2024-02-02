import google.generativeai as genai
import os



os.environ['GOOGLE_API_KEY'] = "AIzaSyD4Lf4VbtMEBzXRoZ2ogHUZpuZLC3Ml6GA"
genai.configure(api_key = os.environ['GOOGLE_API_KEY'])


model = genai.GenerativeModel('gemini-pro')

response = model.generate_content("List 5 planets each with an interesting fact")
print(response.text)