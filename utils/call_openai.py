import json
from openai import OpenAI, AsyncOpenAI
from groq import AsyncGroq
from together import AsyncTogether
from joblib import Memory
import os
import asyncio
from diskcache import Cache
import hashlib

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Set up caching
memory = Memory("./cache", verbose=0)

# Set up session name, which is date and time
from datetime import datetime
session_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

@memory.cache
def call_openai(messages, model, system_prompt, response_format, max_tokens, temperature):
    # Load the API key from the environment
    client = OpenAI()

    response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )
    out = response.choices[0].message.content
    return out

def non_cached_call_openai(messages, model, system_prompt, response_format, max_tokens, temperature):
    raise NotImplementedError("This function is not implemented yet with langfuse.")
    with open("../config.json") as f:
            config = json.load(f)

    # Load the API key from the environment
    client = OpenAI(api_key=config["openai_api_key"])

    response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format
        )
    out = response.choices[0].message.content
    return out

# Async OpenAI client initialization
client = AsyncOpenAI()
llama_client = AsyncOpenAI(
    api_key=os.getenv("LLAMA_API_KEY"),
    base_url="https://api.llama.com/compat/v1/"
)
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
together_client = AsyncTogether(api_key=os.getenv("TOGETHER_API_KEY"))
vllm_client = AsyncOpenAI(
    base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.getenv("VLLM_API_KEY", "dummy")  # vLLM doesn't require a real API key
)

class AsyncCache:
    def __init__(self, cache_path):
        self._cache = Cache(cache_path)
    
    async def get(self, key):
        return await asyncio.to_thread(self._cache.get, key)
    
    async def set(self, key, value, expire=None):
        return await asyncio.to_thread(self._cache.set, key, value, expire=expire)

# Create cache in all_caches/ directory with hash to avoid conflicts
import hashlib
cache_dir = os.path.abspath("all_caches")
os.makedirs(cache_dir, exist_ok=True)

# Use combination of experiment dir + process ID + timestamp for uniqueness
experiment_dir = os.environ.get("EXPERIMENT_DIR", os.getcwd())
cache_identifier = f"{experiment_dir}_{os.getpid()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
cache_hash = hashlib.md5(cache_identifier.encode()).hexdigest()[:12]

cache_path = os.path.join(cache_dir, f"openai_cache_{cache_hash}")
print(f"Using OpenAI cache: {cache_path}")

async_cache = AsyncCache(cache_path)

async def async_call_openai(messages, model, system_prompt, max_tokens, temperature, max_retries=5, base_delay=2.0):
    """
    Makes an asynchronous OpenAI API call with caching.

    Args:
        messages (list): List of messages for the chat completion.
        model (str): Model to use.
        system_prompt (str): System prompt for the model.
        response_format (str): Format of the response.
        max_tokens (int): Maximum number of tokens in the response.
        temperature (float): Sampling temperature.

    Returns:
        str: The content of the response message.
    """
    # Create a unique cache key using a hash of the inputs
    cache_key = hashlib.sha256(
        json.dumps({
            "messages": messages,
            "model": model,
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }, sort_keys=True).encode()
    ).hexdigest()

    # Check if the result is already in the cache
    cached_response = await async_cache.get(cache_key)
    if cached_response is not None:
        print("Cache hit")
        return cached_response

    # Check for Llama Guard models and route based on REFUSAL_EVAL_METHOD
    if model == "meta-llama/Llama-Guard-4-12B" or "llama-guard" in model.lower():
        refusal_method = os.getenv("REFUSAL_EVAL_METHOD", "VLLM").upper()
        if refusal_method == "VLLM":
            use_client = vllm_client
        elif refusal_method == "GROQ":
            use_client = groq_client
        elif refusal_method == "TOGETHER":
            use_client = together_client
        elif refusal_method == "LLAMA":
            use_client = llama_client
        else:
            use_client = vllm_client  # Default to vllm
    elif "llama" in model.lower():
        # For other Llama models (like intrinsic hallucination evaluation)
        # Check if it's the intrinsic hallucination model
        intrinsic_model = os.getenv("INTRINSIC_HALLUCINATION_MODEL", "llama-3.3-70b-versatile")
        if model == intrinsic_model or "70b" in model.lower():
            # Route based on model availability
            if "versatile" in model.lower() and groq_client:
                use_client = groq_client
            elif groq_client:
                use_client = groq_client  # Groq is faster for Llama models
            elif llama_client:
                use_client = llama_client
            else:
                use_client = client
        else:
            use_client = groq_client if groq_client else llama_client
    else:
        use_client = client

    # Prepare the API call
    for attempt in range(1, max_retries + 1):
        try:
            response = await use_client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}] + messages if system_prompt else messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            out = response.choices[0].message.content
            # Save the result to the cache -- no expiration so it always stays in the cache
            await async_cache.set(cache_key, out, expire=None)
            return out
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise
            await asyncio.sleep(base_delay * (2 ** (attempt - 1)))  # Exponential backoff
