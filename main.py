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

openai.api_key = os.getenv('OPENAI_API_KEY') # Replace 'YOUR_OPENAI_API_KEY' with your actual API key
geolocator = Nominatim(user_agent="trip-planner")

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Constants
EXAMPLE_DESTINATIONS = [
    'Ernakulam', 'Fort Kochi', 'Mattancherry', 'Cherai Beach', 'Munnar', 'Thekkady', 'Athirappilly',
    'Kumarakom', 'Marari Beach', 'Alappuzha', 'Thrissur', 'Kottayam', 'Idukki'
]

def generate_prompt(destination, arrival_to, arrival_date, arrival_time, departure_from,
                    departure_date, departure_time, additional_information, **kwargs):
    num_days = (departure_date - arrival_date).days + 1
    return f'''
Prepare a {num_days}-day trip schedule for {destination}, a vibrant city in the Indian state of Kerala, known for its rich cultural heritage, breathtaking backwaters, and diverse culinary delights. Here are the details:

* Arrival To: {arrival_to}
* Arrival Date: {arrival_date}
* Arrival Time: {arrival_time}

* Departure From: {departure_from}
* Departure Date: {departure_date}
* Departure Time: {departure_time}

* Additional Notes: {additional_information}

Include visits to popular attractions like Fort Kochi, Mattancherry Palace, Cherai Beach, and the backwaters of Kumarakom. Recommend opportunities to experience local cuisine, art forms like Kathakali, and shopping for spices and handicrafts.

Also, provide daily route recommendations for exploring the destination while considering factors like traffic, weather, and distance between locations.
'''.strip()


def extract_points_of_interest(itinerary_text):
    pois = []
    doc = nlp(itinerary_text)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            pois.append(ent.text.strip())
    return pois


def display_map(locations):
    m = folium.Map()
    coords = []
    for location in locations:
        try:
            location_data = geolocator.geocode(location)
            if location_data:
                lat, lon = location_data.latitude, location_data.longitude
                coords.append((lat, lon))
                folium.Marker([lat, lon], popup=location).add_to(m)
        except:
            pass
    if len(coords) > 1:
        folium.PolyLine(coords, color="red", weight=5, opacity=0.8).add_to(m)
    components.html(m._repr_html_(), height=500)


def submit():
    # Generate the prompt
    prompt = generate_prompt(**st.session_state)

    # Generate output
    output = openai.Completion.create(
        engine='gpt-3.5-turbo-instruct',
        prompt=prompt,
        temperature=0.45,
        max_tokens=1024
    )

    # Store the generated itinerary
    st.session_state['output'] = output['choices'][0]['text']

    # Split the generated itinerary into individual days
    itinerary = st.session_state['output']
    days = re.split(r'Day \d+:', itinerary)

    num_days = (st.session_state['departure_date'] - st.session_state['arrival_date']).days + 1


    # Display maps for each day
    for i, day in enumerate(days[1:num_days+1], start=1):
        day_itinerary = day.strip()

        st.subheader(f'Day {i} Itinerary:')
        st.write(day_itinerary)

        # Extract points of interest for the current day
        locations = extract_points_of_interest(day_itinerary)

        # Display map with points of interest for the current day
        display_map(locations)


# Initialization
if 'output' not in st.session_state:
    st.session_state['output'] = '--'

st.title('Trippr')
st.subheader('Let us plan your trip!')

with st.form(key='trip_form'):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader('Destination')
        origin = st.text_input('Destination', value=random.choice(EXAMPLE_DESTINATIONS), key='destination')
        st.form_submit_button('Submit', on_click=submit)

    with c2:
        st.subheader('Arrival')

        st.selectbox('Arrival To',
                     ('Airport', 'Train Station', 'Bus Station', 'Ferry Terminal', 'Port', 'Other'),
                     key='arrival_to')
        st.date_input('Arrival Date', value=datetime.now().date() + timedelta(days=1), key='arrival_date')
        st.time_input('Arrival Time', value=datetime.now().time(), key='arrival_time')

    with c3:
        st.subheader('Departure')

        st.selectbox('Departure From',
                     ('Airport', 'Train Station', 'Bus Station', 'Ferry Terminal', 'Port', 'Other'),
                     key='departure_from')
        st.date_input('Departure Date', value=datetime.now().date() + timedelta(days=2), key='departure_date')
        st.time_input('Departure Time', value=datetime.now().time(), key='departure_time')

    st.text_area('Additional Information', height=200,
                 value='I want to visit as many places as possible! (respect time)',
                 key='additional_information')

    st.subheader('Trip Schedule')
    st.write(st.session_state.output)