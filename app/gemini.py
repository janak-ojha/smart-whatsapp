import json
import os
import requests
import logging
import re
import time

logger = logging.getLogger(__name__)

class GeminiLLMClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GeminiLLMClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Gemini Configuration
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.model = "gemini-2.0-flash"
        self.base_url_template = "https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

        if not self.api_key:
            logger.error("Gemini API key is not configured. Set GEMINI_API_KEY environment variable or Django setting.")
            raise ValueError("Gemini API key is required")

        # Construct the URL for the chosen model
        self.generate_url = self.base_url_template.format(model_name=self.model)

    def _gemini_api_call(self, system_instruction, user_prompt, temperature, max_tokens, top_p):
        """
        Helper function to make calls to the Gemini API.
        Constructs the payload, posts to the API endpoint, and returns a plain-text response.
        """
        headers = {
            "Content-Type": "application/json",
        }
        url = f"{self.generate_url}?key={self.api_key}"

        # Gemini expects a list of message-turns. Here we combine system instruction with the user prompt.
        contents = [
            {
                "role": "user",
                "parts": [{"text": f"{system_instruction}\n\nUser Query:\n{user_prompt}"}]
            }
        ]

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": top_p,
            }
        }

        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=(10, 180)  # connect timeout, read timeout
            )
            response.raise_for_status()
            result = response.json()

            # Parse the response and extract the text
            if not result.get('candidates'):
                finish_reason = result.get('promptFeedback', {}).get('blockReason')
                if finish_reason:
                    logger.error(f"Gemini API call blocked or failed. Reason: {finish_reason}")
                    raise ValueError(f"Gemini request failed: {finish_reason}")
                else:
                    logger.error(f"Gemini API response missing 'candidates'. Full response: {result}")
                    raise KeyError("Response from Gemini API is missing 'candidates' field.")

            content = result['candidates'][0].get('content', {})
            parts = content.get('parts', [])
            if not parts or 'text' not in parts[0]:
                logger.error(f"Gemini API response missing 'text' in parts. Full response: {result}")
                raise KeyError("Response from Gemini API is missing 'text' content.")

            reply = parts[0]['text'].strip()
            return reply

        except requests.Timeout as e:
            logger.error(f"Gemini API request timed out during call: {str(e)}")
            raise
        except requests.RequestException as e:
            logger.error(f"Gemini API request failed during call: {str(e)}")
            raise
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Error processing Gemini response: {str(e)}")
            raise

    def _get_outline(self, prompt, context=None):
        """
        Generates a plain-text outline from Gemini to guide the final response.
        """
        system_instruction = (
            "You are a technical planning assistant. Generate an outline for a response to the user's query. "
            "The outline should include 3-5 main sections with 2-3 bullet points each, formatted as plain text."
        )

        if context:
            system_instruction += f"\n\nRelevant Context: {context}"

        logger.info("Fetching response outline from Gemini...")
        try:
            outline = self._gemini_api_call(
                system_instruction=system_instruction,
                user_prompt=f"Create a plain-text outline for responding to this query: {prompt}",
                temperature=0.3,  # Lower temperature for structured output
                max_tokens=400,   # Generous token limit for outline
                top_p=0.9
            )
            logger.info("Successfully generated response outline.")
            # Optional validation: check if the outline contains expected keywords
            if not outline or "Topic" not in outline:
                logger.warning(f"Generated outline may be in unexpected format: {outline}")
                raise ValueError("Outline generation resulted in unexpected format.")
            return outline
        except Exception as e:
            logger.error(f"Error generating outline with Gemini: {str(e)}. Falling back to default.")
            return (
                "Topic Overview:\n- Key points\n"
                "Details:\n- Important details\n"
                "Conclusion:\n- Summary points"
            )

    def _get_full_response(self, prompt, outline, context=None, max_length=800):
        """
        Generates the full plain-text response from Gemini using the outline as a guide.
        """
        domain_instructions = (
            "You are a helpful AI assistant providing technical support and information. "
            "Deliver accurate and helpful answers based on the provided information. "
            "Structure your responses clearly with an introduction, relevant details, and conclusion. "
            "Keep your tone friendly yet professional, and prioritize accuracy and clarity."
        )

        system_instruction = (
            f"Generate a complete plain-text response based on the outline provided. "
            f"Your response should be well-structured, easy to understand via WhatsApp, "
            f"and under {max_length} tokens. If you don't know something, admit it "
            f"rather than inventing information. Use conversational language that works "
            f"well in messaging format."
        )
        if context:
            system_instruction += f"\n\nRelevant Context: {context}"
        system_instruction += f"\n\nOutline:\n{outline}\n\n{domain_instructions}"

        logger.info("Generating full response from Gemini with outline guidance...")
        try:
            full_response = self._gemini_api_call(
                system_instruction=system_instruction,
                user_prompt=prompt,  # The original prompt drives the content
                temperature=0.6,
                max_tokens=max_length,
                top_p=0.9
            )
            logger.info("Successfully generated full response.")
            return full_response
        except Exception as e:
            logger.error(f"Error generating full response with Gemini: {str(e)}")
            raise

    def generate_response(self, prompt, context=None, max_length=800):
        """
        Main method to generate a plain-text response for the given prompt.
        Returns a hardcoded greeting for simple cases or uses Gemini for full responses.
        """
        overall_start = time.perf_counter()

        basic_greetings = {"hi", "hii", "hello", "hey", "hlo", "h", "hh", "hiii", "helloo", "helo", "hilo", "hellooo"}
        normalized_prompt = prompt.strip().lower()

        if normalized_prompt in basic_greetings:
            hardcoded_response = (
                "Hello! How can I help you today with Presage Insights? "
                "I can assist with predictive maintenance, IoT sensor data, or analytics questions."
            )
            logger.info("Returning hardcoded greeting response.")
            return hardcoded_response

        try:
            outline_response = self._get_outline(prompt, context)
            full_response = self._get_full_response(prompt, outline_response, context, max_length)
            overall_elapsed = time.perf_counter() - overall_start
            logger.info(f"Total generate_response time: {overall_elapsed:.2f} seconds.")
            return full_response
        except requests.Timeout:
            logger.error("Gemini API request timed out")
            return "Sorry, the response is taking too long. Please try again later."
        except requests.RequestException as e:
            error_detail = ""
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_detail = f" - {error_data.get('error', {}).get('message', e.response.text)}"
                except json.JSONDecodeError:
                    error_detail = f" - Status Code: {e.response.status_code}"
            logger.error(f"Gemini API request failed: {str(e)}{error_detail}")
            return "I apologize, but I'm having trouble processing your request. Please try again later."
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected response format from Gemini API: {str(e)}")
            return "I encountered an error while generating a response (unexpected format). Please try again."
        except Exception as e:
            logger.error(f"Unexpected error in generate_response: {str(e)}", exc_info=True)
            return "An unexpected error occurred. Please try again later."

    def query_llm(self, messages, temperature=0.5, max_tokens=800, top_p=0.9):
        """
        Direct query interface for Gemini LLM.
        Accepts a list of messages in the format [{'role': 'user', 'content': ...}, ...]
        and returns a plain-text response.
        """
        logger.info("Querying Gemini LLM directly...")
        contents = []
        system_instruction = ""
        current_user_prompt = ""

        if messages and messages[0]['role'] == 'system':
            system_instruction = messages[0]['content']
            messages = messages[1:]

        if messages and messages[0]['role'] == 'user':
            current_user_prompt = messages[0]['content']
            combined_first_prompt = f"{system_instruction}\n\n{current_user_prompt}".strip()
            contents.append({"role": "user", "parts": [{"text": combined_first_prompt}]})
        elif system_instruction:
            contents.append({"role": "user", "parts": [{"text": system_instruction}]})
        else:
            logger.error("query_llm requires at least a user message.")
            return "Error: No user message provided for query."

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": top_p,
            }
        }
        headers = {"Content-Type": "application/json"}
        url = f"{self.generate_url}?key={self.api_key}"

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=(10, 180))
            response.raise_for_status()
            result = response.json()

            if not result.get('candidates'):
                finish_reason = result.get('promptFeedback', {}).get('blockReason')
                if finish_reason:
                    logger.error(f"Gemini direct query blocked. Reason: {finish_reason}")
                    return f"Request failed due to safety settings or other issue: {finish_reason}"
                else:
                    logger.error("Gemini direct query response missing 'candidates'.")
                    return "Error: Unexpected response format from Gemini (missing candidates)."

            content = result['candidates'][0].get('content', {})
            parts = content.get('parts', [])
            if not parts or 'text' not in parts[0]:
                logger.error("Gemini direct query response missing 'text' in parts.")
                return "Error: Unexpected response format from Gemini (missing text)."

            reply = parts[0]['text'].strip()
            logger.info("Received direct response from Gemini LLM.")
            return reply

        except requests.Timeout:
            logger.error("Gemini LLM direct query timed out")
            return "Request timed out. Please try again."
        except requests.RequestException as e:
            error_detail = ""
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_detail = f" - {error_data.get('error', {}).get('message', e.response.text)}"
                except json.JSONDecodeError:
                    error_detail = f" - Status Code: {e.response.status_code}"
            logger.error(f"Gemini LLM direct query failed: {str(e)}{error_detail}")
            return "Request failed. Please check logs."
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Unexpected response format or processing error in direct query: {str(e)}")
            return "Unexpected error occurred while parsing the LLM response."
        except Exception as e:
            logger.error(f"General error in query_llm: {str(e)}", exc_info=True)
            return "Unexpected error. Please try again later."