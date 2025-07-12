from pytube import Playlist
import yt_dlp
from moviepy import AudioFileClip
import streamlit as st
import zipfile 
import io 
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def reduce_filename(filename: str) -> str:
    """
    Reduce filename to 3-4 clear words using LangChain 0.3
    
    Args:
        filename (str): The original filename to reduce
        openai_api_key (str): OpenAI API key (optional, can use environment variable)
    
    Returns:
        str: Reduced filename with 3-4 clear words
    """
    
    
    
    # Create ChatOpenAI model
    # llm = ChatOpenAI(
    #     model="gpt-3.5-turbo",
    #     temperature=0,  # Keep it consistent
    #     api_key=openai_api_key
    # )

    llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    # max_tokens=None,
    # timeout=None,
    # max_retries=2,
    api_key = GEMINI_API_KEY
    
)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        You are a filename expert. Your job is to reduce long filenames to SHORT FORMS with underscores.
        
        Rules:
        1. Use short forms of words (abbreviations)
        2. Connect words with underscores (_)
        3. MUST USE EXACTLY 3-4 WORDS MAXIMUM - NO MORE THAN 4 WORDS!
        4. Make it lowercase
        5. Remove unnecessary words like "the", "a", "an", "of", "to", "introduction", "fundamentals"
        6. Use common abbreviations
        7. Pick only the MOST IMPORTANT words from the filename
        8. If filename is very long, choose only 3-4 main topics
        
        Example:
        Input: "English US Course Level 1 Unit 1"
        Output: "eng_lvl1_unit1"
        
        Input: "Advanced Python Programming Tutorial Chapter 5"
        Output: "adv_python_ch5"
        
        Input: "Machine Learning Data Science Project"
        Output: "ml_data_proj"
        
        Input: "Data Science Project Analysis Report Final Version"
        Output: "data_sci_proj"
        
        Input: "Web Development HTML CSS JavaScript Complete Guide"
        Output: "web_dev_guide"
        
        IMPORTANT: Never use more than 4 words! Choose only the most important ones.
        
        Now reduce this filename:
        Input: "{filename}"
        Output: """
    )
    
    # Create output parser
    output_parser = StrOutputParser()
    
    # Create chain using LangChain 0.3 syntax
    chain = prompt | llm | output_parser
    
    # Run the chain
    result = chain.invoke({"filename": filename})
    
    # Clean up result (remove extra spaces and quotes, make lowercase)
    reduced_name = result.strip().strip('"').strip("'").lower()
    
    return reduced_name



