#!/usr/bin/env python3
"""
Gemini API Generative Testing Script
Tests various generative capabilities of the Gemini API
"""

import os
import json
import time
import requests
from typing import Dict, Any, List

class GeminiAPITester:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        # Updated model names - try different models if one doesn't work
        self.models = [
            "gemini-1.5-flash",
            "gemini-1.5-pro", 
            "gemini-2.0-flash-exp",
            "gemini-pro"  # fallback
        ]
        self.current_model = self.models[0]  # Start with the most common one
        
    def make_request(self, prompt: str, temperature: float = 0.7) -> Dict[str, Any]:
        """Make a request to the Gemini API"""
        url = f"{self.base_url}/{self.current_model}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": temperature,
                "topK": 1,
                "topP": 1,
                "maxOutputTokens": 2048,
                "stopSequences": []
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return {"error": f"Model {self.current_model} not found. Status: {e.response.status_code}"}
            else:
                return {"error": f"HTTP Error {e.response.status_code}: {str(e)}"}
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def extract_text(self, response: Dict[str, Any]) -> str:
        """Extract generated text from API response"""
        try:
            if "error" in response:
                return f"Error: {response['error']}"
            
            candidates = response.get("candidates", [])
            if not candidates:
                return "No candidates found in response"
            
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            
            if not parts:
                return "No parts found in response"
            
            return parts[0].get("text", "No text found")
        except Exception as e:
            return f"Error extracting text: {str(e)}"
    
    def test_model_availability(self) -> None:
        """Test which models are available"""
        print("🔍 Testing Model Availability")
        print("=" * 50)
        
        simple_prompt = "Hello, how are you?"
        
        for model in self.models:
            self.current_model = model
            print(f"Testing model: {model}")
            response = self.make_request(simple_prompt)
            
            if "error" not in response:
                print(f"✅ {model} - Working!")
                result = self.extract_text(response)
                print(f"   Response: {result[:100]}...")
                print()
                return  # Use this working model
            else:
                print(f"❌ {model} - {response['error'][:100]}...")
        
        print("❌ No working models found!")
        print()

    def test_basic_generation(self) -> None:
        """Test basic text generation"""
        print("🧪 Testing Basic Text Generation")
        print("=" * 50)
        
        prompt = "Explain quantum computing in simple terms."
        response = self.make_request(prompt)
        result = self.extract_text(response)
        
        print(f"Prompt: {prompt}")
        print(f"Response: {result[:200]}...")
        print()
    
    def test_creative_writing(self) -> None:
        """Test creative writing capabilities"""
        print("✨ Testing Creative Writing")
        print("=" * 50)
        
        prompt = "Write a short story about a robot who discovers emotions for the first time."
        response = self.make_request(prompt, temperature=0.9)
        result = self.extract_text(response)
        
        print(f"Prompt: {prompt}")
        print(f"Response: {result[:300]}...")
        print()
    
    def test_code_generation(self) -> None:
        """Test code generation"""
        print("💻 Testing Code Generation")
        print("=" * 50)
        
        prompt = "Write a Python function that calculates the fibonacci sequence up to n numbers."
        response = self.make_request(prompt, temperature=0.3)
        result = self.extract_text(response)
        
        print(f"Prompt: {prompt}")
        print(f"Response: {result[:400]}...")
        print()
    
    def test_question_answering(self) -> None:
        """Test Q&A capabilities"""
        print("❓ Testing Question Answering")
        print("=" * 50)
        
        questions = [
            "What is the capital of France?",
            "How does photosynthesis work?",
            "What are the benefits of renewable energy?"
        ]
        
        for question in questions:
            response = self.make_request(question)
            result = self.extract_text(response)
            print(f"Q: {question}")
            print(f"A: {result[:150]}...")
            print("-" * 30)
    
    def test_summarization(self) -> None:
        """Test text summarization"""
        print("📝 Testing Summarization")
        print("=" * 50)
        
        long_text = """
        Artificial intelligence (AI) is intelligence demonstrated by machines, 
        as opposed to natural intelligence displayed by animals including humans. 
        Leading AI textbooks define the field as the study of "intelligent agents": 
        any system that perceives its environment and takes actions that maximize 
        its chance of achieving its goals. Some popular accounts use the term 
        "artificial intelligence" to describe machines that mimic "cognitive" 
        functions that humans associate with the human mind, such as "learning" 
        and "problem solving". AI research has been highly successful in 
        developing effective techniques for solving a wide range of problems, 
        from game playing to medical diagnosis.
        """
        
        prompt = f"Summarize this text in 2-3 sentences: {long_text}"
        response = self.make_request(prompt)
        result = self.extract_text(response)
        
        print(f"Original text length: {len(long_text)} characters")
        print(f"Summary: {result}")
        print()
    
    def test_translation(self) -> None:
        """Test language translation"""
        print("🌍 Testing Translation")
        print("=" * 50)
        
        prompt = "Translate 'Hello, how are you today?' to French, Spanish, and German."
        response = self.make_request(prompt)
        result = self.extract_text(response)
        
        print(f"Prompt: {prompt}")
        print(f"Response: {result}")
        print()
    
    def test_reasoning(self) -> None:
        """Test logical reasoning"""
        print("🧠 Testing Reasoning")
        print("=" * 50)
        
        prompt = """
        If all roses are flowers, and some flowers are red, can we conclude that some roses are red? 
        Explain your reasoning step by step.
        """
        
        response = self.make_request(prompt)
        result = self.extract_text(response)
        
        print(f"Logic Problem: {prompt.strip()}")
        print(f"Response: {result}")
        print()
    
    def run_all_tests(self) -> None:
        """Run all generative tests"""
        print("🚀 Starting Gemini API Generative Testing")
        print("=" * 60)
        print()
        
        # First test which model works
        self.test_model_availability()
        
        # List test method names so we can safely handle missing methods
        test_names = [
            'test_basic_generation',
            'test_creative_writing',
            'test_code_generation',
            'test_question_answering',
            'test_summarization',
            'test_translation',
            'test_reasoning'
        ]

        available_tests = [name for name in test_names if hasattr(self, name) and callable(getattr(self, name))]
        total = len(available_tests)
        for i, name in enumerate(available_tests, 1):
            print(f"Test {i}/{total}: {name}")
            test_fn = getattr(self, name)
            try:
                test_fn()
                time.sleep(1)  # Rate limiting - be respectful to the API
            except Exception as e:
                print(f"❌ {name} failed with error: {str(e)}")
                print()
        
        print("🎉 All tests completed!")

def main():
    # Set your API key here or as an environment variable
    api_key = "AIzaSyAi4STQMt6vrhwVVCGnfFXilJL-33OFrmQ"
    
    # You can also get it from environment variable for better security
    # Prefer configured AgentConfig primary key
    try:
        from server import config
        if hasattr(config, 'gemini_api_keys') and config.gemini_api_keys:
            api_key = config.gemini_api_keys[0]
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("❌ Error: API key not found. Please set GEMINI_API_KEY environment variable or configure gemini_api_keys in server.AgentConfig.")
        return
    
    tester = GeminiAPITester(api_key)
    tester.run_all_tests()

if __name__ == "__main__":
    main()