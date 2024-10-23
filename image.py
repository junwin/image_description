import os
import argparse
import json
import base64
import requests
import subprocess
from PIL import Image, JpegImagePlugin, PngImagePlugin

# Load OpenAI credentials from environment variables
credential_path = os.getenv("CREDENTIAL_PATH", "/home/your_user_id/credential")
with open(os.path.join(credential_path, "oaicred.json"), "r") as config_file:
    config_data = json.load(config_file)

openai_api_key = os.getenv("OPENAI_API_KEY", config_data.get("openai_api_key"))


def sanitize_description(description):
    """Sanitize the description to avoid issues with command line execution."""
    sanitized_description = description.replace('"', "'").replace('\n', ' ')
    return sanitized_description

def encode_image(image_path):
    """Encode an image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def clean_markdown(text):
    """Remove markdown symbols and clean up the text."""
    text = text.replace('**', '').strip()  # Remove bold markers
    return text

def parse_keywords(keywords_section):
    """Parse the keywords section to extract a list of keywords."""
    keywords = []
    for line in keywords_section.splitlines():
        line = line.strip()
        if line.startswith('-'):
            keyword = line.lstrip('-').strip().replace(' ', '')  # Remove spaces from keywords
            keywords.append(keyword)
    return keywords


def parse_iptc_data(iptc_data):
    """#
    Parse IPTC binary data and return a dictionary of IPTC properties.

    :param iptc_data: A byte array containing IPTC binary data.
    :returns: A dictionary containing IPTC properties.
    """
    #hexdump.hexdump(iptc_data, 'print')
    results = {}
    tag_start = b'\x1c\x02'
    pos = 0

    while True:
        start = iptc_data.find(tag_start, pos)
        if start < 0:
            break

        typeMajor = int.from_bytes(iptc_data[start+1:start+2], byteorder='big')
        typeMinor = int.from_bytes(iptc_data[start+2:start+3], byteorder='big')
        tag_type = str(typeMajor) + ":" + str(typeMinor)
        tag_len = int.from_bytes(iptc_data[start+3:start+5], byteorder='big')
        tag_end = start + 5 + tag_len
        
        tag_value = iptc_data[start+5:tag_end].decode('utf-8')
        # print(start, tag_type, tag_len, tag_end, tag_value)
        tag_key = IPTC_TAG_TYPES.get(str(tag_type), 'Unknown')
        if tag_key in results:
            if not isinstance(results[tag_key], list):
                results[tag_key] = [results[tag_key]]
            results[tag_key].append(tag_value)
        else:
            results[tag_key] = tag_value
        pos = tag_end
    return results


def showImageIptcMeta(file_path):
    """Extract existing IPTC metadata."""
    with Image.open(file_path) as img:
        iptc = img.info.get('photoshop', {})
        iptcInfo = iptc.get(1028, b'')
        results = parse_iptc_data(iptcInfo)
        title = results.get('Object Attribute', '')
        description = results.get('Caption/Abstract Writer', '')
        keywords = results.get('Keywords', [])
        processed_keywords = [''.join([word.capitalize() for word in keyword.split()]) for keyword in keywords]
        return title, description, processed_keywords
    

IPTC_TAG_TYPES = {
    "2:0": "Record Version",
    "2:3": "Object Type",
    "2:5": "Object Attribute",
    "2:7": "Object Name",
    "2:10": "Edit Status",
    "2:12": "Editorial Update",
    "2:15": "Urgency",
    "2:20": "Keyword",
    "2:22": "Category",
    "2:25": "Keywords",
    "2:26": "Location",
    "2:27": "City",
    "2:30": "Caption/Abstract",
    "2:40": "Instructions",
    "2:55": "Date Created",
    "2:60": "Time Created",
    "2:62": "Digital Creation Date/Time",
    "2:63": "Originating Program",
    "2:80": "Byline",
    "2:85": "Byline Title",
    "2:90": "City",
    "2:92": "Sublocation",
    "2:95": "State/Province",
    "2:100": "Country/Primary Location Name",
    "2:101": "Country/Primary Location Code",
    "2:103": "Original Transmission Reference",
    "2:105": "Headline",
    "2:110": "Credit",
    "2:115": "Source",
    "2:116": "Copyright Notice",
    "2:118": "Contact",
    "2:120": "Caption/Abstract Writer",
    "2:122": "Rasterized Caption",
    "2:130": "Content Location Code",
    "2:131": "Content Location Name",
    "2:135": "ICC Profile",
    "2:150": "Writer/Editor",
    "2:151": "Image Type",
    "2:184": "Job ID",
    "2:185": "Master Document ID",
    "2:186": "Short Document ID",
    "2:187": "Unique Document ID",
    "2:188": "Owner ID",
    "2:200": "Object Preview File Format",
    "2:201": "Object Preview File Format Version",
    "2:202": "Object Preview Data"
}

IPTC_TAG_TYPES_INV = {v: k for k, v in IPTC_TAG_TYPES.items()}


def generate_openai_description_and_keywords(image_path, existing_title, existing_description, existing_keywords):
    """Generate descriptions for the visually challenged and an enhanced description along with keywords using OpenAI."""
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    # Include existing metadata in the prompt and request a JSON response
    prompt = f"""
    You are an assistant tasked with enhancing the metadata of an image. Here's the existing information:
    
    Title: {existing_title}
    Description: {existing_description}
    Keywords: {', '.join(existing_keywords)}
    
    Please return the following in plain JSON format without any additional formatting or code blocks:
    1) A description of this image suitable for someone who is visually challenged.
    2) An enhanced description discussing the artistic rationale behind the image based on the provided title and description - keep the text first person, simple and not too wordy
    3) A set of keywords suitable for social media, listed as an array.

    Respond in the following JSON format:
    {{
        "visually_challenged_description": "A description suitable for the visually challenged.",
        "enhanced_description": "Your enhanced description discussing the artistic rationale.",
        "keywords": ["keyword1", "keyword2", "keyword3"]
    }}
    """

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    print("DEBUG: Sending API request to OpenAI...")
    #print(json.dumps(payload, indent=2))
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    # Debug: Print the entire response for inspection
    print("DEBUG: Full API Response:")
    print(json.dumps(response.json(), indent=2))

    response_data = response.json()

    # Extracting the JSON content
    try:
        content = response_data['choices'][0]['message']['content']
        
        # Debug: Print the extracted content
        print("DEBUG: Extracted Content:")
        print(content)
        
        # Parse the JSON response
        response_json = json.loads(content)
        
        # Extract the descriptions and keywords from the JSON
        visually_challenged_description = response_json.get("visually_challenged_description", "No description available")
        enhanced_description = response_json.get("enhanced_description", "No enhanced description available")
        keywords = response_json.get("keywords", [])
        
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"DEBUG: Error parsing JSON response: {e}")
        visually_challenged_description = 'No description available'
        enhanced_description = 'No enhanced description available'
        keywords = []

    return visually_challenged_description, enhanced_description, keywords

import os

def process_image(file_path):
    # Determine the output .json file name
    json_file_path = file_path.rsplit('.', 1)[0] + '.json'

    # Skip processing if the .json file already exists
    if os.path.exists(json_file_path):
        print(f"Skipping {file_path} as {json_file_path} already exists.")
        return

    # Extract existing metadata
    existing_title, existing_description, existing_keywords = showImageIptcMeta(file_path)
    
    print(f"Existing Title: {existing_title}")
    print(f"Existing Description: {existing_description}")
    print(f"Existing Keywords: {existing_keywords}")
    
    # Analyze the image with OpenAI to generate the necessary descriptions and keywords
    visually_challenged_description, enhanced_description, keywords = generate_openai_description_and_keywords(
        file_path, 
        existing_title, 
        existing_description, 
        existing_keywords
    )

    # Sanitize the descriptions to avoid issues with command line execution
    visually_challenged_description = sanitize_description(visually_challenged_description)
    enhanced_description = sanitize_description(enhanced_description)
    
    print(f"Visually Challenged Description: {visually_challenged_description}")
    print(f"Enhanced Description: {enhanced_description}")
    print(f"Keywords: {keywords}")

    # Merge existing and new keywords, ensure lowercase and remove duplicates
    merged_keywords = list(set((existing_keywords + keywords)))
    merged_keywords = [kw.lower() for kw in merged_keywords]

    # Prepare hashtags as a single string, ensure lowercase and remove duplicates
    hashtags = ' '.join([f"#{kw}" for kw in merged_keywords])

    # Debug: Print merged keywords
    print(f"Merged Keywords: {merged_keywords}")
    
    # Write the title, descriptions, and keywords to the .json file
    metadata = {
        "title": existing_title,
        "visually_challenged_description": visually_challenged_description,
        "enhanced_description": enhanced_description,
        "keywords": merged_keywords,
        "hashtags": hashtags
    }

    with open(json_file_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Wrote metadata to {json_file_path}")

def process_json_and_update_image(json_file_path):
    """Read metadata from a .json file and update the corresponding image using ExifTool."""
    with open(json_file_path, 'r') as f:
        metadata = json.load(f)
    
    # Determine the corresponding image file path
    image_file_path_jpg = json_file_path.rsplit('.', 1)[0] + '.jpg'
    image_file_path_png = json_file_path.rsplit('.', 1)[0] + '.png'
    if os.path.exists(image_file_path_jpg):
        image_file_path = image_file_path_jpg
    elif os.path.exists(image_file_path_png):
        image_file_path = image_file_path_png
    else:
        print(f"No corresponding image found for JSON {json_file_path}")
        return

    # Update metadata using ExifTool, setting each keyword separately
    cmd_base = [
        'exiftool',
        f'-ImageDescription={metadata.get("enhanced_description", "")}',
        f'-Caption-Abstract={metadata.get("enhanced_description", "")}',
        f'-Description={metadata.get("enhanced_description", "")}',
        f'-ObjectName={metadata.get("title", "")}',
        f'-Title={metadata.get("title", "")}'
    ]

    for keyword in metadata.get("keywords", []):
        keyword_no_spaces = keyword.replace(' ', '')
        cmd_base.append(f'-keywords+={keyword_no_spaces}')

    cmd_base.append('-overwrite_original')
    cmd_base.append(image_file_path)

    # Log the command line being used
    print(f"Running command: {' '.join(cmd_base)}")

    # Execute the command and capture the output
    result = subprocess.run(cmd_base, capture_output=True, text=True)

    # Log the output from ExifTool
    print("ExifTool Output:")
    print(result.stdout)
    if result.stderr:
        print("ExifTool Error:")
        print(result.stderr)

def process_folder(directory):
    for file_name in os.listdir(directory):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(directory, file_name)
            process_image(file_path)
        elif file_name.endswith('.json'):
            json_file_path = os.path.join(directory, file_name)
            process_json_and_update_image(json_file_path)

def main():
    parser = argparse.ArgumentParser(description='Process images in a directory with OpenAI and IPTC Meta')
    parser.add_argument('work_dir', type=str, help='the directory containing images to process')
    args = parser.parse_args()
    work_dir = args.work_dir
    if not os.path.isdir(work_dir):
        print(f"The provided directory {work_dir} does not exist.")
        exit(1)
    process_folder(work_dir)

if __name__ == "__main__":
    main()