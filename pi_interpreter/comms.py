import requests
import json
from datetime import datetime
import pytz
import time

access_token = ''
master_host = "pblsv-ec2.thpt.tk:8080"

def login(username, password):
    global access_token
    
    url = f"http://{master_host}/login" 
    payload = {
        'username': username,
        'password': password
    }
    headers = {}
    
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()  
        
        data = response.json()
        
        print(data)
        
        access_token = data.get('access_token')  # Extracts the access token from the response
        
        return access_token
    except requests.RequestException as e:
        return str(e)  # Returns the error occurred during the request

def report_fall_detected(fall_count):

    global access_token
    
    url = f"http://{master_host}/report_fall_detected"
    
    #timezone = pytz.timezone('Etc/GMT-7')
    #current_time = datetime.now(pytz.utc).astimezone(timezone).strftime("%Y-%m-%d %H:%M:%S")
    
    payload = json.dumps({
        "time_report": int(time.time()) + 25200, # GMT+7 alignment (yes i know it sucks)
        "sequence_label_as_fall_count": fall_count
    })
    print(payload)
    headers = {
        'Authorization': f'Bearer {access_token}', 
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status() 
        data = response.json() 
            
        print(data)
        
        turn_on_beeper_count = data.get('turn_on_beeper_count', 0)  
        need_video_upload = data.get('need_video_upload', False)  
        video_upload_ticket = data.get('video_upload_ticket', '')

        return turn_on_beeper_count, need_video_upload, video_upload_ticket

    except requests.RequestException as e:
        return str(e)
        
def upload_video(filename, video_upload_ticket):
    url = f"http://{master_host}/upload_video/{video_upload_ticket}"  
    headers = {
        'Authorization': f'Bearer {access_token}' 
    }

    files = [
        ('file', (filename, open(filename, 'rb'), 'application/octet-stream'))
    ]
    payload = {}
    
    try:
        response = requests.request("POST", url, headers=headers, data=payload, files=files)
        response.raise_for_status() 

        data = response.json()
              
        return data.get('message') == "File uploaded successfully"
    except requests.RequestException as e:
        print(f"An error occurred: {str(e)}")
        return False
    finally:
        files[0][1][1].close()
