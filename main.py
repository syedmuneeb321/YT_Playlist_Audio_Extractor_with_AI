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
        9. If filename is unit,part ..etc you must add at the end
        
        Example:
        Input: "[name] unit 1"
        Output: "[abrivation]_unit_1"
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




zip_buffer = io.BytesIO()

def download_audio_from_playlist(playlist_url, start, end, download_path, audio_path):
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    if not os.path.exists(audio_path):
        os.makedirs(audio_path)

    playlist = Playlist(playlist_url)
    videos = playlist.video_urls
    st.write(f"Total videos in playlist: {len(videos)}")

    if end > len(videos):
        end = len(videos)

    ydl_opts = {
        'outtmpl': os.path.join(download_path, '%(title)s.%(ext)s'),
        'format': 'bestaudio'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for i in range(start-1, end):
            video_url = videos[i]
            st.write(f"🔻 Downloading audio {i+1}: {video_url}")

            try:
                ydl.download([video_url])
            except Exception as e:
                st.error(f"❌ Failed to download video {i+1}: {e}")

    for file in os.listdir(download_path):
        if file.endswith(".webm") or file.endswith(".m4a"):
            input_path = os.path.join(download_path, file)
            output_name = reduce_filename(os.path.splitext(file)[0]) + ".mp3"
            output_path = os.path.join(audio_path, output_name)

            st.write(f"🎵 Converting {file} to mp3...")

            try:
                clip = AudioFileClip(input_path)
                clip.write_audiofile(output_path)
                clip.close()

                os.remove(input_path)
                st.write(f"✅ Deleted original file: {file}")

            except Exception as e:
                st.error(f"❌ Failed to convert {file}: {e}")

    st.success("🎉 All audios converted and cleaned.")
    return True
# -----------------------
# 📱 Streamlit UI
# -----------------------

st.title("🎶 YouTube Playlist Audio Downloader")

playlist_url = st.text_input("Enter YouTube Playlist URL:")
start_video = st.number_input("Start Video Number:", min_value=1, value=1)
end_video = st.number_input("End Video Number:", min_value=1, value=3)

download_folder = "Downloaded_Audio"
mp3_folder = "Converted_Audios"

if st.button("🎬 Start Download"):
    if playlist_url.strip() == "":
        st.error("Please enter a playlist URL.")
    elif start_video > end_video:
        st.error("Start video number must be less than or equal to End video number.")
    else:
        done = download_audio_from_playlist(playlist_url, start_video, end_video, download_folder, mp3_folder)
        st.success("✅ Done! Check your Converted_Audios folder.")
        if done:
            with zipfile.ZipFile(zip_buffer,'w') as zipf:
                for file in os.listdir(mp3_folder):
                    if file.endswith(".mp3"):
                        file_path = os.path.join(mp3_folder,file)
                        zipf.write(file_path,arcname=file)
            zip_buffer.seek(0)

            is_donwloaded = st.download_button(
                label="⬇️ Download All Audios as ZIP",
                data=zip_buffer,
                file_name="converted_mp3.zip",
                mime="application/zip"
            )

            # If button clicked, delete files
            if is_donwloaded:
                for file in os.listdir(mp3_folder):
                    if file.endswith(".mp3"):
                        os.remove(os.path.join(mp3_folder, file))
                st.success("✅ All audio files deleted after download.")


