import os
import random
from datetime import datetime, timedelta
import openai
import streamlit as st
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
import folium

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
geolocator = Nominatim(user_agent="trip-planner")

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
    # Filter out POIs that are not present in the set of possible POIs
    
    # Split the itinerary text into lines
    lines = itinerary_text.split('\n')
    
    # Initialize a list to store the extracted POIs
    pois = []
    
    # Iterate through each line in the itinerary text
    for line in lines:
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        
        # Check if the line is not empty and is in the set of possible POIs
        if line and line in possible_pois:
            pois.append(line)
    
    return pois



def display_map(locations):
    # Create a folium map object
    m = folium.Map()

    # Add markers for each location
    for location in locations:
        try:
            # Geocode the location
            location_data = geolocator.geocode(location)
            if location_data:
                lat, lon = location_data.latitude, location_data.longitude
                folium.Marker([lat, lon], popup=location).add_to(m)
        except:
            pass

    # Display the map
    st.write(m)


def submit():
    prompt = generate_prompt(**st.session_state)

    # generate output
    output = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0.45,
        top_p=1,
        frequency_penalty=2,
        presence_penalty=0,
        max_tokens=1024
    )

    st.session_state['output'] = output['choices'][0]['text']

    # Extract points of interest
    locations = extract_points_of_interest(output['choices'][0]['text'])
    
    # Display map with points of interest
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
        st.form_submit_button('Submit', on_click=submit)

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
