import os
import random
from datetime import datetime, timedelta
import openai
import streamlit as st
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
import folium
import csv
import spacy
import streamlit.components.v1 as components
import re

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
geolocator = Nominatim(user_agent="trip-planner")

# Load spaCy NER model
nlp = spacy.load("en_core_web_trf")  # Use the larger 'en_core_web_trf' model

# Define example destinations and random destination
example_destinations = [
    'Izmir', 'Istanbul', 'Ankara', 'Paris', 'London', 'New York', 'Tokyo', 'Sydney', 'Hong Kong',
    'Singapore', 'Warsaw', 'Mexico City', 'Palermo'
]
random_destination = random.choice(example_destinations)

now_date = datetime.now()

# round to nearest 15 minutes
now_date = now_date.replace(minute=now_date.minute // 15 * 15, second=0, microsecond=0)

# split into date and time objects
now_time = now_date.time()
now_date = now_date.date() + timedelta(days=1)


def generate_prompt(destination, arrival_to, arrival_date, arrival_time, departure_from,
                    departure_date, departure_time, additional_information, **kwargs):
    return f'''
Prepare trip schedule for {destination}, based on the following information:

* Arrival To: {arrival_to}
* Arrival Date: {arrival_date}
* Arrival Time: {arrival_time}

* Departure From: {departure_from}
* Departure Date: {departure_date}
* Departure Time: {departure_time}

* Additional Notes: {additional_information}
'''.strip()


def extract_points_of_interest(itinerary_text, possible_pois):
    # Extract points of interest from the itinerary text based on a set of possible POIs
    # Combine Geospatial Analysis, Named Entity Recognition (NER), and Keyword Matching
    
    # Initialize a list to store the extracted POIs
    pois = []
    
    # Use spaCy NER to identify named entities (locations) in the text
    doc = nlp(itinerary_text)
    for entity in doc.ents:
        if entity.label_ == "GPE":  # Geopolitical Entity (Location)
            poi = entity.text.strip()
            if poi.lower() in possible_pois:
                pois.append(poi)
    
    return pois


def load_possible_pois(csv_file_path):
    # Load possible POIs from a CSV file
    pois = set()
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            poi = row.get('POI')
            if poi:
                pois.add(poi.strip().lower())  # Remove leading/trailing whitespace and convert to lowercase
    
    return pois


def display_map(locations):
    # Create a folium map object
    m = folium.Map()
    coords = []

    # Add markers for each location
    for location in locations:
        try:
            # Geocode the location
            location_data = geolocator.geocode(location)
            if location_data:
                lat, lon = location_data.latitude, location_data.longitude
                coords.append((lat, lon))
                folium.Marker([lat, lon], popup=location).add_to(m)
        except:
            pass
    if len(coords) > 1:
        folium.PolyLine(coords, color="red", weight=5, opacity=0.8).add_to(m)
    m.save('map.html')
    with open('map.html', 'r', encoding='utf-8') as f:
        html = f.read()
    components.html(html, height=500)


def generate_itinerary():
    # Generate the prompt
    prompt = generate_prompt(**st.session_state)

    # Generate output
    output = openai.Completion.create(
        engine='gpt-3.5-turbo-instruct',
        prompt=prompt,
        temperature=0.45,
        top_p=1,
        frequency_penalty=2,
        presence_penalty=0,
        max_tokens=1024
    )

    # Store the generated itinerary
    st.session_state['output'] = output['choices'][0]['text']

    # Load possible POIs from CSV
    possible_pois = load_possible_pois('possible_pois.csv')
    
    # Split the generated itinerary into individual days
    itinerary = st.session_state['output']
    days = re.split(r'Day \d+:', itinerary)

    # Display itinerary and maps for each day
    for i, day in enumerate(days, start=1):
        if i == 0:
            continue  # Skip the first item, which is the text before the first day

        day_number = i
        day_itinerary = day.strip()

        st.subheader(f'Day {day_number} Itinerary:')
        st.write(day_itinerary)

        # Extract points of interest for the current day
        locations = extract_points_of_interest(day_itinerary, possible_pois)

        # Display map with points of interest for the current day
        display_map(locations)


# Initialization
if 'output' not in st.session_state:
    st.session_state['output'] = '--'

st.title('GPT-3 Trip Scheduler')
st.subheader('Let us plan your trip!')

with st.form(key='trip_form'):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader('Destination')
        origin = st.text_input('Destination', value=random_destination, key='destination')
        st.form_submit_button('Submit', on_click=generate_itinerary)

    with c2:
        st.subheader('Arrival')

        st.selectbox('Arrival To',
                     ('Airport', 'Train Station', 'Bus Station', 'Ferry Terminal', 'Port', 'Other'),
                     key='arrival_to')
        st.date_input('Arrival Date', value=now_date, key='arrival_date')
        st.time_input('Arrival Time', value=now_time, key='arrival_time')

    with c3:
        st.subheader('Departure')

        st.selectbox('Departure From',
                     ('Airport', 'Train Station', 'Bus Station', 'Ferry Terminal', 'Port', 'Other'),
                     key='departure_from')
        st.date_input('Departure Date', value=now_date + timedelta(days=1), key='departure_date')
        st.time_input('Departure Time', value=now_time, key='departure_time')

    st.text_area('Additional Information', height=200,
                 value='I want to visit as many places as possible! (respect time)',
                 key='additional_information')

    st.subheader('Trip Schedule')
    st.write(st.session_state.output)