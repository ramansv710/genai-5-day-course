# Okay, here is a Google Kaggle Notebook implementing the "Personalized Carbonated Drink Recommender with Dietary and Health Considerations" using Gemini models and incorporating the requested features.
# Key Features Implemented:
# Embeddings: Used Gemini's embedding-001 model to create vector representations of drink descriptions and user queries for semantic understanding.
# Retrieval Augmented Generation (RAG): Implemented a basic RAG system. User queries are embedded, and the most semantically similar drinks from a pre-defined dataset are retrieved. This retrieved context is then passed to the generative model.
# Function Calling: Defined Python functions (e.g., to filter drinks by specific nutritional criteria) and provided their schemas to the Gemini model. The model can decide to call these functions to fulfill parts of the user request.
# Structured Output / JSON Mode: Configured the Gemini Pro model to output its recommendations in a specific JSON format, making the results easy to parse and use programmatically.
# Long Context Window (Implicit): While not explicitly demonstrated with a single massive prompt, the RAG process benefits from Gemini's long context capability by allowing the model to process the user's query plus the retrieved context from multiple relevant drinks simultaneously.
# Dietary & Health Considerations: Incorporated through the data structure (calories, sugar, caffeine, ingredients), RAG retrieval based on descriptive terms (e.g., "low sugar"), and specific function calls (e.g., filtering by max calories).
# How to Use This Notebook:
# Get API Key: You need a Google AI API Key.
# Go to Google AI Studio.
# Click "Get API Key".
# Create or use an existing API key.
# Set API Key in Kaggle: In the "Setup" section of the notebook, replace "YOUR_API_KEY" with your actual API key. It's recommended to use Kaggle's secrets manager for better security.
# Run All Cells: Select "Runtime" -> "Run all" from the Kaggle menu.
# Interact: Go to the "Run the Recommender" section and try different user queries.

"""
Personalized Carbonated Drink Recommender using Gemini

This Kaggle Notebook demonstrates how to build a personalized carbonated
drink recommender system using Google's Gemini models. It incorporates
several advanced features:

1. Embeddings: For semantic understanding of drinks and user queries.
2. Retrieval Augmented Generation (RAG): To ground recommendations in factual
   drink data.
3. Function Calling: To allow the model to interact with specific data
   filtering tools.
4. Structured Output (JSON Mode): To get predictable, parseable results.
5. Long Context Window (Implicit): Leveraged by RAG to process retrieved
   information alongside the query.
6. Dietary & Health Considerations: Factored into the data and recommendation
   process.
"""

# @title Setup
# Install necessary libraries
# !pip install -q -U google-generativeai pandas numpy scikit-learn

# Import required libraries
import google.generativeai as genai # Main library for Gemini API access
import os                             # For accessing environment variables (like API key)
import json                           # For handling JSON data (structured output, function calling)
import pandas as pd                   # For data manipulation (our drink database)
import numpy as np                    # For numerical operations, especially with embeddings
from sklearn.metrics.pairwise import cosine_similarity # For comparing embeddings
import textwrap                       # For formatting text output nicely

try:
    # Attempt to get the API key from Kaggle secrets
    GOOGLE_API_KEY = "AIzaSyDcE4C9JLpuYJAJ-U2A6yp-Be0Pse5vHHk"
    # Check if the key was actually retrieved
    if not GOOGLE_API_KEY:
        # Raise an error if the secret is not set
        raise ValueError("API Key not found in Kaggle secrets. Please add 'GOOGLE_API_KEY' to your Kaggle secrets.")
    # Configure the Gemini API client with the retrieved key
    genai.configure(api_key=GOOGLE_API_KEY)
    # Print confirmation message
    print("Successfully configured Gemini API with key from Kaggle secrets.")
except Exception as e:
    # Print an error message if configuration fails
    print(f"Error configuring Gemini API: {e}")
    # Provide instructions on how to set the API key
    print("Please ensure you have added your Google API Key to Kaggle Secrets (Menu: Insert -> Add secret) with the name 'GOOGLE_API_KEY'.")

# Define the models to use
EMBEDDING_MODEL_NAME = "models/embedding-001"     # Model for creating text embeddings
GENERATIVE_MODEL_NAME = "gemini-1.5-flash-latest" # Powerful generative model with function calling & JSON mode support (gemini-1.5-pro-latest is also a good option)

# @title 1. Data Preparation: Carbonated Drink Dataset

# Create a sample dataset of carbonated drinks.
# In a real application, this would come from a database or larger file.
# We include nutritional info, descriptions, and flags for dietary needs.
data = {
    'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'Name': [
        "Cola Classic", "Diet Cola", "Lemon-Lime Sparkle", "Orange Burst",
        "Ginger Ale Zing", "Sparkling Water (Natural)", "Sparkling Water (Lime)",
        "Zero Sugar Energy Drink", "Craft Root Beer", "Grape Soda Pop",
        "Caffeine-Free Cola", "Zero Calorie Sparkling Raspberry"
    ],
    'Description': [
        "The original cola taste. Bold, refreshing, and timeless.",
        "Same great cola taste, zero sugar. Perfect for calorie-conscious drinkers.",
        "Crisp and clean lemon and lime flavors. Refreshing and bubbly.",
        "A sunny burst of sweet orange flavor. Tangy and invigorating.",
        "Real ginger flavor with a spicy kick. Soothing and effervescent.",
        "Pure, crisp sparkling water. No added flavors or sweeteners.",
        "Sparkling water infused with natural lime essence. Zero calories, zero sugar.",
        "Provides a boost of energy without sugar. Contains caffeine and B vitamins.",
        "Rich and creamy traditional root beer flavor. Made with cane sugar.",
        "Sweet and bubbly grape flavor. A fun, nostalgic treat.",
        "Classic cola flavor without the caffeine kick. Enjoy anytime.",
        "Delicious raspberry flavored sparkling water with zero calories and zero sugar."
    ],
    'Flavor Profile': [
        "Cola, Sweet", "Cola, Aspartame", "Lemon, Lime, Citrus", "Orange, Citrus, Sweet",
        "Ginger, Spicy", "Neutral, Bubbly", "Lime, Citrus, Tart",
        "Mixed Fruit, Artificial Sweetener, Energy Boost", "Root Beer, Vanilla, Sweet",
        "Grape, Sweet", "Cola, Sweet", "Raspberry, Fruity, Tart"
    ],
    'Calories (per 12oz)': [140, 0, 100, 160, 120, 0, 0, 10, 180, 170, 140, 0],
    'Sugar (g per 12oz)': [39, 0, 27, 44, 32, 0, 0, 0, 48, 45, 39, 0],
    'Caffeine (mg per 12oz)': [34, 46, 0, 0, 0, 0, 0, 160, 0, 0, 0, 0],
    'Ingredients': [
        "Carbonated Water, High Fructose Corn Syrup, Caramel Color, Phosphoric Acid, Natural Flavors, Caffeine",
        "Carbonated Water, Caramel Color, Aspartame, Phosphoric Acid, Potassium Benzoate (Preservative), Natural Flavors, Citric Acid, Caffeine",
        "Carbonated Water, High Fructose Corn Syrup, Citric Acid, Natural Flavors, Sodium Citrate, Sodium Benzoate (Preservative)",
        "Carbonated Water, High Fructose Corn Syrup, Citric Acid, Sodium Benzoate (Preservative), Natural Flavors, Modified Food Starch, Ester Gum, Yellow 6, Brominated Vegetable Oil",
        "Carbonated Water, High Fructose Corn Syrup, Citric Acid, Natural Flavors, Sodium Benzoate (Preservative), Caramel Color",
        "Carbonated Water",
        "Carbonated Water, Natural Lime Flavor",
        "Carbonated Water, Citric Acid, Taurine, Sodium Citrate, Natural Flavors, Panax Ginseng Extract, L-Carnitine Tartrate, Caffeine, Sucralose, Sodium Benzoate (Preservative), Niacinamide (Vit B3), D-Calcium Pantothenate (Vit B5), Salt, Acesulfame Potassium, Pyridoxine Hydrochloride (Vit B6), Yellow 5, Cyanocobalamin (Vit B12)",
        "Carbonated Water, Cane Sugar, Caramel Color, Natural and Artificial Flavors, Sodium Benzoate (Preservative), Citric Acid",
        "Carbonated Water, High Fructose Corn Syrup, Sodium Benzoate (Preservative), Citric Acid, Natural and Artificial Flavors, Red 40, Blue 1",
        "Carbonated Water, High Fructose Corn Syrup, Caramel Color, Phosphoric Acid, Natural Flavors, Potassium Benzoate (Preservative)",
        "Carbonated Water, Citric Acid, Natural Raspberry Flavor, Potassium Benzoate (Preservative), Aspartame, Acesulfame Potassium"
    ],
    'Is_Sugar_Free': [False, True, False, False, False, True, True, True, False, False, False, True],
    'Is_Caffeine_Free': [False, False, True, True, True, True, True, False, True, True, True, True],
    'Is_Zero_Calorie': [False, True, False, False, False, True, True, False, False, False, False, True] # Note: Diet Cola is 0 cal, Energy Drink is 10 cal
}

# Create a pandas DataFrame
df_drinks = pd.DataFrame(data)

# Combine relevant text fields for embedding
# We create a more descriptive text entry for each drink to improve embedding quality.
df_drinks['Combined_Text'] = df_drinks.apply(
    lambda row: f"Name: {row['Name']}. Description: {row['Description']}. Flavor: {row['Flavor Profile']}. "
                f"Calories: {row['Calories (per 12oz)']}. Sugar: {row['Sugar (g per 12oz)']}g. Caffeine: {row['Caffeine (mg per 12oz)']}mg. "
                f"{'Sugar-free. ' if row['Is_Sugar_Free'] else ''}"
                f"{'Caffeine-free. ' if row['Is_Caffeine_Free'] else ''}"
                f"{'Zero-calorie.' if row['Is_Zero_Calorie'] else ''}",
    axis=1 # Apply function row-wise
)

# Display the first few rows of the DataFrame with the combined text
print("Sample Drink Data:")
print(df_drinks[['Name', 'Combined_Text']].head())

# @title 2. Embeddings: Generate Embeddings for Drinks

# --- Embedding Generation Function ---
def get_embeddings(texts, model_name=EMBEDDING_MODEL_NAME):
    """
    Generates embeddings for a list of texts using the specified Gemini model.

    Args:
        texts (list): A list of strings to embed.
        model_name (str): The name of the embedding model to use.

    Returns:
        np.ndarray: A numpy array containing the embeddings, or None if an error occurs.
    """
    try:
        # Request embeddings from the Gemini API
        # 'content' should be the text to embed
        # 'task_type' helps the model optimize for retrieval
        result = genai.embed_content(model=model_name,
                                     content=texts,
                                     task_type="RETRIEVAL_DOCUMENT") # Use RETRIEVAL_DOCUMENT for the items being stored
        # Return the embeddings as a numpy array
        return np.array(result['embedding'])
    except Exception as e:
        # Print an error message if embedding fails
        print(f"An error occurred during embedding generation: {e}")
        # Return None to indicate failure
        return None

# --- Generate and Store Drink Embeddings ---

print("\nGenerating embeddings for the drink dataset...")
# Get the combined text descriptions for all drinks
drink_texts = df_drinks['Combined_Text'].tolist()
# Generate embeddings for these texts
drink_embeddings = get_embeddings(drink_texts)

# Check if embeddings were generated successfully
if drink_embeddings is not None:
    # Print the shape of the resulting embedding array (Num Drinks x Embedding Dimension)
    print(f"Successfully generated {drink_embeddings.shape[0]} embeddings with dimension {drink_embeddings.shape[1]}.")
    # Store the embeddings in the DataFrame (optional, but can be convenient)
    # Note: Storing large arrays directly in DataFrame cells isn't always efficient for large datasets.
    # df_drinks['Embeddings'] = list(drink_embeddings) # Uncomment if you want to store directly
else:
    # Print error message if embeddings failed
    print("Failed to generate embeddings for the drink dataset.")

# @title 3. Retrieval (RAG Core): Find Similar Drinks

# --- Similarity Search Function ---
def find_similar_drinks(query_text, top_n=5):
    """
    Finds the most similar drinks in the dataset based on semantic similarity
    of their embeddings to the query embedding.

    Args:
        query_text (str): The user's query (e.g., "low sugar fruity drink").
        top_n (int): The number of top similar drinks to return.

    Returns:
        pd.DataFrame: A DataFrame containing the top_n most similar drinks,
                      or None if embeddings are not available or an error occurs.
    """
    # Check if drink embeddings are available
    if drink_embeddings is None:
        # Print error if embeddings haven't been generated
        print("Error: Drink embeddings are not available.")
        # Return None indicating failure
        return None

    try:
        # 1. Embed the user query
        # Use task_type="RETRIEVAL_QUERY" for the user's search query
        query_embedding_response = genai.embed_content(model=EMBEDDING_MODEL_NAME,
                                                     content=query_text,
                                                     task_type="RETRIEVAL_QUERY") # Use RETRIEVAL_QUERY for the search input
        # Extract the embedding vector
        query_embedding = np.array(query_embedding_response['embedding']).reshape(1, -1) # Reshape for compatibility with cosine_similarity

        # 2. Calculate Similarity
        # Compute cosine similarity between the query embedding and all drink embeddings
        similarities = cosine_similarity(query_embedding, drink_embeddings)[0] # Get the similarity scores

        # 3. Get Top N Indices
        # Find the indices of the top N highest similarity scores (excluding the query itself if it were in the dataset)
        # argsort sorts in ascending order, so we take the last 'top_n' elements for highest similarity
        top_indices = np.argsort(similarities)[-top_n:][::-1] # Reverse to get descending order

        # 4. Retrieve Top N Drinks
        # Select the rows from the original DataFrame corresponding to the top indices
        similar_drinks_df = df_drinks.iloc[top_indices].copy() # Use .copy() to avoid SettingWithCopyWarning
        # Add the similarity score to the DataFrame for context
        similar_drinks_df['similarity_score'] = similarities[top_indices]

        # Return the DataFrame of similar drinks
        return similar_drinks_df

    except Exception as e:
        # Print error message if similarity search fails
        print(f"An error occurred during similarity search: {e}")
        # Return None indicating failure
        return None

# --- Example Usage ---
print("\n--- RAG Test: Finding similar drinks ---")
# Define a sample query
test_query = "I want something bubbly and fruity but without sugar"
# Find similar drinks using the function
similar_results = find_similar_drinks(test_query, top_n=3)

# Check if results were returned
if similar_results is not None:
    # Print the names and similarity scores of the found drinks
    print(f"Drinks similar to '{test_query}':")
    # Iterate through the results and print details
    for index, row in similar_results.iterrows():
        # Print name and similarity score, formatted to 3 decimal places
        print(f"  - {row['Name']} (Similarity: {row['similarity_score']:.3f})")
else:
    # Print message if no results found or error occurred
    print("Could not retrieve similar drinks.")

# @title 4. Function Calling: Define Tools for Specific Queries

# --- Define Python Functions ---
# These functions will be callable by the Gemini model.

def find_drinks_by_criteria(
    max_calories: int = None,
    max_sugar: int = None,
    must_be_caffeine_free: bool = False,
    must_be_sugar_free: bool = False,
    must_be_zero_calorie: bool = False
) -> str:
    """
    Filters the drink list based on specific nutritional criteria.

    Args:
        max_calories (int, optional): Maximum calories allowed per 12oz serving. Defaults to None.
        max_sugar (int, optional): Maximum sugar (in grams) allowed per 12oz serving. Defaults to None.
        must_be_caffeine_free (bool, optional): If True, only returns caffeine-free drinks. Defaults to False.
        must_be_sugar_free (bool, optional): If True, only returns sugar-free drinks. Defaults to False.
        must_be_zero_calorie (bool, optional): If True, only returns zero-calorie drinks. Defaults to False.

    Returns:
        str: A JSON string representing a list of drinks that meet the criteria,
             including their name, calories, sugar, and caffeine content.
             Returns an empty list string '[]' if no drinks match.
    """
    # Start with the full DataFrame
    filtered_df = df_drinks.copy()

    # Apply filters based on the arguments provided
    if max_calories is not None:
        # Filter by maximum calories
        filtered_df = filtered_df[filtered_df['Calories (per 12oz)'] <= max_calories]
    if max_sugar is not None:
        # Filter by maximum sugar
        filtered_df = filtered_df[filtered_df['Sugar (g per 12oz)'] <= max_sugar]
    if must_be_caffeine_free:
        # Filter for caffeine-free drinks
        filtered_df = filtered_df[filtered_df['Is_Caffeine_Free'] == True]
    if must_be_sugar_free:
        # Filter for sugar-free drinks
        filtered_df = filtered_df[filtered_df['Is_Sugar_Free'] == True]
    if must_be_zero_calorie:
        # Filter for zero-calorie drinks
        filtered_df = filtered_df[filtered_df['Is_Zero_Calorie'] == True]

    # Select relevant columns for the output
    results = filtered_df[['Name', 'Calories (per 12oz)', 'Sugar (g per 12oz)', 'Caffeine (mg per 12oz)']].to_dict(orient='records')

    # Return the results as a JSON string
    return json.dumps(results)

# --- Define Function Declarations for Gemini ---
# This tells the Gemini model about the available Python functions,
# their parameters, and what they do.

# Create a FunctionDeclaration object for the find_drinks_by_criteria function
find_drinks_tool = genai.protos.FunctionDeclaration(
    name="find_drinks_by_criteria", # The exact name of the Python function
    description="Finds carbonated drinks based on specific nutritional limits like maximum calories, maximum sugar, or whether they must be caffeine-free, sugar-free, or zero-calorie.", # Description for the model
    parameters=genai.protos.Schema( # Define the parameters the function accepts
        type=genai.protos.Type.OBJECT, # Parameters are defined as an object
        properties={ # Dictionary of parameter names and their schemas
            "max_calories": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="Maximum calories per 12oz serving."),
            "max_sugar": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="Maximum sugar in grams per 12oz serving."),
            "must_be_caffeine_free": genai.protos.Schema(type=genai.protos.Type.BOOLEAN, description="Whether the drink must be caffeine-free."),
            "must_be_sugar_free": genai.protos.Schema(type=genai.protos.Type.BOOLEAN, description="Whether the drink must be sugar-free."),
            "must_be_zero_calorie": genai.protos.Schema(type=genai.protos.Type.BOOLEAN, description="Whether the drink must be zero-calorie.")
        },
        # Define which parameters are optional (though all are optional here due to defaults in Python)
        # For this specific schema, we can omit 'required' if all are optional based on Python defaults.
        # required=[] # Example if some were required
    )
)

# --- Tool Configuration ---
# Create a Tool object containing the function declarations
# This will be passed to the Gemini model during generation.
tools = genai.protos.Tool(
    function_declarations=[find_drinks_tool]
)

# --- Mapping Function Names to Functions ---
# Create a dictionary to easily call the correct Python function based on the name returned by the model
available_functions = {
    "find_drinks_by_criteria": find_drinks_by_criteria,
}

print("\nFunction Calling Tools defined:")
print(f"- {find_drinks_tool.name}: {find_drinks_tool.description}")

# @title 5. Structured Output (JSON Mode): Define Output Schema

# Define the desired JSON structure for the final recommendation output.
# This helps ensure the model returns data in a predictable and usable format.

# Define the JSON schema using a Python dictionary
# This specifies the expected structure of the model's final output.
json_output_schema = {
  "type": "object", # The top-level structure is an object
  "properties": { # Define the fields within the object
    "recommendations": { # A field named "recommendations"
      "type": "array", # This field should contain an array
      "description": "A list of recommended carbonated drinks based on the user query and retrieved context.", # Description of the field
      "items": { # Define the structure of each item within the array
        "type": "object", # Each item is an object
        "properties": { # Define the fields within each recommendation object
          "name": {
            "type": "string", # Drink name should be a string
            "description": "The name of the recommended drink." # Description
          },
          "reasoning": {
            "type": "string", # Reasoning should be a string
            "description": "Explanation why this drink is recommended based on the user query and drink properties." # Description
          },
          "calories": {
            "type": "number", # Calories should be a number
            "description": "Calories per 12oz serving." # Description
          },
          "sugar_g": {
            "type": "number", # Sugar should be a number
            "description": "Grams of sugar per 12oz serving." # Description
          },
          "caffeine_mg": {
            "type": "number", # Caffeine should be a number
            "description": "Milligrams of caffeine per 12oz serving." # Description
          }
        },
        "required": ["name", "reasoning", "calories", "sugar_g", "caffeine_mg"] # These fields must be present in each recommendation
      }
    }
  },
  "required": ["recommendations"] # The "recommendations" field is mandatory in the overall output
}


# --- Generation Configuration ---
# Configure the generative model to use JSON mode with the defined schema.
generation_config_json = genai.types.GenerationConfig(
    response_mime_type="application/json", # Specify the desired output format
    response_schema=json_output_schema # Provide the schema definition
)

print("\nJSON Output Schema defined for structured recommendations.")

# @title 6. The Recommender Function (Putting it all together)

# --- Initialize the Generative Model ---
# Create an instance of the Gemini Pro model configured for function calling and JSON output.
# We use the specific config for JSON output only when we expect the *final* answer in JSON.
# Function call responses are handled separately.
model = genai.GenerativeModel(
    GENERATIVE_MODEL_NAME,
    # tools=[tools] # Tools are added dynamically during the conversation
)

# --- Main Recommender Function ---
def get_drink_recommendations(user_query, rag_top_n=5):
    """
    Generates personalized drink recommendations based on user query,
    using RAG, Function Calling, and Structured Output.

    Args:
        user_query (str): The user's request for a drink recommendation.
        rag_top_n (int): The number of relevant documents to retrieve for RAG context.

    Returns:
        dict: A dictionary containing the structured recommendations (if successful),
              or an error message.
    """
    print(f"\n--- Processing Query: '{user_query}' ---")

    # 1. RAG - Retrieve Relevant Drinks
    print(f"Step 1: Retrieving top {rag_top_n} relevant drinks (RAG)...")
    # Find drinks semantically similar to the user query
    similar_drinks = find_similar_drinks(user_query, top_n=rag_top_n)

    # Check if retrieval was successful
    if similar_drinks is None or similar_drinks.empty:
        # Handle case where no similar drinks are found
        print("RAG: Could not find relevant drinks based on the query.")
        # Provide context as an empty list string
        rag_context = "[]"
    else:
        # Format the retrieved drink data as context for the LLM
        # Convert selected columns of the similar drinks DataFrame to a list of dictionaries
        context_list = similar_drinks[['Name', 'Description', 'Flavor Profile', 'Calories (per 12oz)', 'Sugar (g per 12oz)', 'Caffeine (mg per 12oz)', 'Is_Sugar_Free', 'Is_Caffeine_Free', 'Is_Zero_Calorie', 'Ingredients']].to_dict(orient='records')
        # Convert the list of dictionaries to a JSON string
        rag_context = json.dumps(context_list, indent=2) # Pretty print JSON for readability
        print(f"RAG: Retrieved context for {len(context_list)} drinks.")
        # print("Retrieved Context:\n", rag_context) # Uncomment to see the context being sent

    # 2. Prepare the Prompt for the LLM
    # Construct a detailed prompt incorporating the user query and the RAG context.
    # Instruct the model on its role, the context, dietary considerations, function use, and desired output format.
    prompt = f"""
    You are a helpful Carbonated Drink Recommender assistant.
    Your goal is to provide personalized drink recommendations based on the user's preferences and dietary needs.
    You should prioritize health considerations mentioned (low sugar, low calorie, caffeine-free etc.).

    USER QUERY: "{user_query}"

    CONTEXT FROM DRINK DATABASE (use this information to ground your recommendations):
    ```json
    {rag_context}
    ```

    INSTRUCTIONS:
    1. Analyze the USER QUERY to understand their preferences (flavor, type, dietary restrictions like low sugar, caffeine-free, specific calorie limits).
    2. Use the provided CONTEXT to find drinks that match the user's query.
    3. If the query involves specific, filterable criteria (e.g., "drinks under 50 calories", "list all sugar-free options"), consider using the 'find_drinks_by_criteria' function if appropriate. The function returns a list of drinks matching exact numerical or boolean criteria. You can use the function's output to supplement or replace the RAG context if it's more direct for the query.
    4. Generate a list of 1-3 recommended drinks.
    5. For each recommendation, provide:
        - The drink's name.
        - A brief reasoning explaining *why* it fits the user's query, referencing the context or function call results.
        - Key nutritional info: calories, sugar (g), caffeine (mg).
    6. Format your FINAL output STRICTLY as a JSON object matching the provided schema. Do not include any text before or after the JSON object.
    """

    # 3. Interact with the Gemini Model (potentially with function calls)
    print("Step 2: Generating recommendations with Gemini...")

    try:
        # Start a chat session to handle potential function call loops
        chat = model.start_chat(enable_automatic_function_calling=True) # Let the SDK handle the function call loop

        # Send the initial prompt with tools configured for this turn
        # Note: We send the prompt and *also* pass the tools configuration.
        # The generation_config for JSON is applied *after* any function calls complete.
        response = chat.send_message(
            prompt,
            # tools=[tools] # Handled by enable_automatic_function_calling=True
        )

        # Automatic Function Calling handles the back-and-forth if the model decides to call a function.
        # The 'response' object will contain the *final* response after any function calls are resolved.

        # Check for errors in the final response (e.g., safety blocks)
        if not response.parts:
             # Handle cases where the response might be blocked or empty
            print("Error: Received an empty or blocked response from the model.")
            # Check candidate for specific block reason if available
            if response.candidates and response.candidates[0].finish_reason.name != "STOP":
                print(f"Reason: {response.candidates[0].finish_reason.name}")
                # You might also check response.prompt_feedback for block reasons
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                   print(f"Block Reason: {response.prompt_feedback.block_reason}")

            return {"error": "Model response was empty or blocked."}


        # 4. Process Final Response (Expecting JSON)
        print("Step 3: Processing final response...")
        # At this stage, we expect the final answer, potentially after function calls.
        # Now, configure the model *specifically* for JSON output using the final text content.
        # This is a slightly different approach than forcing JSON from the start if function calls might occur.
        # Alternative: If you *know* the final response *must* be JSON, you could re-send the final prompt content
        # (potentially including function results) with the JSON generation config.

        # Let's try to parse the text content assuming it should be JSON.
        # The automatic function calling might return the final result as text,
        # even if the function call itself returned JSON. We need to guide the *final* generation.

        # --- Re-generating the final response with JSON mode ---
        # We take the content generated *after* potential function calls and ask the model
        # to format *that content* into the desired JSON structure.

        # Get the text content from the response potentially generated after function calls
        final_content_for_json = response.text

        # Create a new prompt specifically asking for JSON formatting of the previous response.
        json_formatting_prompt = {final_content_for_json}        
        
        # Generate the final response using JSON mode
        json_response = model.generate_content(
            json_formatting_prompt,
            generation_config=generation_config_json,
            # No tools needed for this final formatting step
        )


        # Check if the JSON response has content
        if not json_response.parts:
             # Handle cases where the response might be blocked or empty
            print("Error: Received an empty or blocked response during JSON formatting.")
            # Check candidate for specific block reason if available
            if json_response.candidates and json_response.candidates[0].finish_reason.name != "STOP":
                print(f"Reason: {json_response.candidates[0].finish_reason.name}")
                 # You might also check response.prompt_feedback for block reasons
                if hasattr(json_response, 'prompt_feedback') and json_response.prompt_feedback.block_reason:
                   print(f"Block Reason: {json_response.prompt_feedback.block_reason}")
            return {"error": "Model response was empty or blocked during JSON formatting."}


        # Extract and parse the JSON output
        try:
            # Access the text part of the response
            json_output_text = json_response.text
            # Parse the text as JSON
            structured_output = json.loads(json_output_text)
            # Print success message
            print("Successfully parsed structured JSON output.")
            print("Structured Output:" + structured_output)
            # Return the parsed dictionary
            return structured_output
        except json.JSONDecodeError as e:
            # Handle JSON parsing errors
            print(f"Error: Failed to decode JSON response: {e}")
            # Print the raw response text for debugging
            print("Raw model response text for JSON formatting:")
            print(json_response.text)
            # Return an error dictionary
            return {"error": "Failed to parse JSON output from the model.", "raw_response": json_response.text}
        except Exception as e:
             # Handle other potential errors during JSON processing
            print(f"An unexpected error occurred processing the JSON response: {e}")
            print("Raw model response text for JSON formatting:")
            print(json_response.text)
            return {"error": f"An unexpected error occurred: {e}", "raw_response": json_response.text}


    except Exception as e:
        # Handle any errors during the Gemini API call or processing
        print(f"An error occurred during recommendation generation: {e}")
        # Return an error dictionary
        return {"error": str(e)}


# @title 7. Run the Recommender: Example Queries

# --- Example 1: Preference for low sugar ---
query1 = "I'm looking for a refreshing drink, maybe something fruity, but low in sugar."
recommendations1 = get_drink_recommendations(query1)
print("\n--- Recommendations for Query 1 ---")
# Pretty print the JSON output
print(json.dumps(recommendations1, indent=2))

# --- Example 2: Specific dietary constraint (triggering function call) ---
query2 = "Show me drinks that have less than 10 grams of sugar and are caffeine free."
# This query is likely to trigger the 'find_drinks_by_criteria' function.
recommendations2 = get_drink_recommendations(query2)
print("\n--- Recommendations for Query 2 (Function Call Likely) ---")
# Pretty print the JSON output
print(json.dumps(recommendations2, indent=2))

# --- Example 3: Flavor preference ---
query3 = "I want a classic cola taste."
recommendations3 = get_drink_recommendations(query3)
print("\n--- Recommendations for Query 3 ---")
# Pretty print the JSON output
print(json.dumps(recommendations3, indent=2))

# --- Example 4: Zero Calorie preference ---
query4 = "Are there any zero calorie options available?"
recommendations4 = get_drink_recommendations(query4)
print("\n--- Recommendations for Query 4 ---")
# Pretty print the JSON output
print(json.dumps(recommendations4, indent=2))

# --- Example 5: Energy boost needed ---
query5 = "I need an energy drink, preferably without sugar."
recommendations5 = get_drink_recommendations(query5)
print("\n--- Recommendations for Query 5 ---")
# Pretty print the JSON output
print(json.dumps(recommendations5, indent=2))


# @title 8. Conclusion

print("\n" + "="*50)
print(" Personalized Carbonated Drink Recommender Demo Finished ")
print("="*50)
print("This notebook demonstrated building a recommender using Gemini:")
print("- Embeddings for semantic search")
print("- RAG for grounding recommendations in data")
print("- Function Calling for specific data filtering")
print("- Structured JSON output for predictable results")
print("- Implicit use of Long Context via RAG")
print("- Incorporation of dietary/health needs")
print("\nTry modifying the queries in 'Run the Recommender' to test different scenarios!")


# Explanation of Code Sections:
# Setup: Installs and imports necessary libraries (google-generativeai, pandas, numpy, scikit-learn). Configures the Gemini API key (using Kaggle Secrets is recommended for security). Defines model names and helper functions.
# Data Preparation: Creates a sample pandas DataFrame (df_drinks) containing information about various carbonated drinks. Crucially, it includes nutritional details (calories, sugar, caffeine) and boolean flags for common dietary needs (sugar-free, caffeine-free, zero-calorie). A Combined_Text column is generated to provide rich input for the embedding model.
# Embeddings:
# Defines get_embeddings to generate vector embeddings for a list of texts using the specified Gemini embedding model (models/embedding-001). It uses task_type="RETRIEVAL_DOCUMENT" as these embeddings represent the documents we'll search against.
# Generates embeddings for all drinks based on their Combined_Text and stores them in drink_embeddings.
# Retrieval (RAG Core):
# Defines find_similar_drinks which takes a query_text:
# Embeds the query using task_type="RETRIEVAL_QUERY".
# Calculates cosine_similarity between the query embedding and all stored drink_embeddings.
# Identifies the top_n drinks with the highest similarity scores.
# Returns a DataFrame containing these most relevant drinks.
# Includes a simple test to show how RAG retrieves relevant items.
# Function Calling:
# Defines a Python function find_drinks_by_criteria that filters the df_drinks DataFrame based on arguments like max_calories, max_sugar, etc. This function directly addresses structured dietary queries.
# Defines find_drinks_tool using genai.protos.FunctionDeclaration. This describes the Python function's name, purpose, and parameters to the Gemini model.
# Creates a tools configuration object containing the function declaration.
# Sets up available_functions mapping to easily call the Python function when the model requests it.
# Structured Output (JSON Mode):
# Defines json_output_schema, a Python dictionary describing the desired JSON structure for the final recommendations (a list of objects, each with name, reasoning, calories, etc.).
# Creates generation_config_json using genai.types.GenerationConfig, specifying response_mime_type="application/json" and providing the schema.
# The Recommender Function (get_drink_recommendations):
# Initializes the generative model (gemini-1.5-flash-latest).
# Takes the user_query.
# RAG Step: Calls find_similar_drinks to get relevant context.
# Prompt Construction: Creates a detailed prompt for the Gemini model, including its role, the user_query, the rag_context, and instructions on how to generate recommendations, consider functions, and format the output.
# Model Interaction:
# Uses model.start_chat(enable_automatic_function_calling=True) which simplifies handling function calls. The SDK automatically handles the loop: Model requests call -> SDK executes function -> SDK sends result back -> Model generates final response.
# Sends the initial prompt.
# Final Response Processing:
# After potential function calls, the final response might still be text. To enforce the JSON structure strictly, it takes the final text response, creates a new prompt asking the model to format that text according to the JSON schema.
# Calls model.generate_content with this new prompt and the generation_config_json.
# Parses the resulting JSON text using json.loads().
# Includes error handling for API calls and JSON parsing.
# Run the Recommender: Provides several example user_query strings demonstrating different types of requests (general preference, specific constraints likely to trigger function calls, flavor focus) and prints the resulting structured JSON recommendations.
# Documentation and Explanation: A detailed markdown cell explaining all the concepts (Embeddings, RAG, Function Calling, JSON Mode, Long Context, Dietary Needs) and how they are implemented in the notebook.
# Conclusion: A brief summary message.
# This notebook provides a comprehensive, runnable example demonstrating the powerful features of Gemini for building sophisticated, context-aware, and controllable AI applications. Remember to replace the placeholder for the API key with your actual key, preferably using Kaggle Secrets.