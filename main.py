import os
import random
from datetime import datetime, timedelta
import openai
import streamlit as st
from dotenv import load_dotenv
import re
import pandas as pd
from geopy.distance import geodesic
import requests
import streamlit.components.v1 as components
import math
from geopy.geocoders import Nominatim
import spacy
from locations import LOCATIONS
load_dotenv()
streamlit_style = """
            <style>
              @import url('https://fonts.googleapis.com/css2?family=Lato:ital,wght@0,100;0,300;0,400;0,700;1,100&display=swap');

              .hotel-bold {
                font-weight: 600;
              }

              .hotel-font {
                font-size: 20px;
                background-color: #e6f9ff;
              }

              label.css-1p2iens.effi0qh3{
                font-size: 18px;
              }

              p{
                font-size: 18px;
              }
              li{
                font-size: 18px;
              }        
              #MainMenu{
                visibility: hidden;
              }   
              button.css-135zi6y.edgvbvh9{
                font-size: 18px;
                font-weight: 600;
              }
            </style>
            """

st.markdown(streamlit_style, unsafe_allow_html=True)

st.image('./assets/Group.png')

openai.api_key = os.getenv('OPENAI_API_KEY')  # Replace with your actual API key
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
Prepare a {num_days}-day trip schedule for {destination}, Here are the details:

* Arrival To: {arrival_to}
* Arrival Date: {arrival_date}
* Arrival Time: {arrival_time}

* Departure From: {departure_from}
* Departure Date: {departure_date}
* Departure Time: {departure_time}

* Additional Notes: {additional_information}
'''.strip()
def extract_locations(text):
    locations = []
    for location, pois in LOCATIONS.items():
        if location.lower() in text.lower():
            locations.append(location)
            for poi in pois:
                if poi.lower() in text.lower():
                    locations.append(poi)
    return list(set(locations))

def generate_google_maps_link(location_route, loc_df):
    location_route_names = [loc_df[loc_df['Latitude'] == lat]['Place_Name'].values[0].replace(' ', '+')
                            for lat, lon in location_route]
    gmap_search = 'https://www.google.com/maps/dir/+'
    gmap_places = gmap_search + '/'.join(location_route_names) + '/'
    return gmap_places

def tsp_solver(data_model, iterations=1000, temperature=10000, cooling_rate=0.95):
    def distance(point1, point2):
        return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

    num_locations = data_model['num_locations']
    locations = [(float(lat), float(lng)) for lat, lng in data_model['locations']]

    # Handle the case when there is only one location or no locations
    if num_locations == 1:
        return locations
    elif num_locations == 0:
        return []

    # Randomly generate a starting solution
    current_solution = list(range(num_locations))
    random.shuffle(current_solution)

    # Compute the distance of the starting solution
    current_distance = 0
    for i in range(num_locations):
        current_distance += distance(locations[current_solution[i-1]], locations[current_solution[i]])

    # Initialize the best solution as the starting solution
    best_solution = current_solution
    best_distance = current_distance

    # Simulated Annealing algorithm
    for i in range(iterations):
        # Compute the temperature for this iteration
        current_temperature = temperature * (cooling_rate ** i)

        # Generate a new solution by swapping two random locations
        new_solution = current_solution.copy()
        j, k = random.sample(range(num_locations), 2)
        new_solution[j], new_solution[k] = new_solution[k], new_solution[j]

        # Compute the distance of the new solution
        new_distance = 0
        for i in range(num_locations):
            new_distance += distance(locations[new_solution[i-1]], locations[new_solution[i]])

        # Decide whether to accept the new solution
        delta = new_distance - current_distance
        if delta < 0 or random.random() < math.exp(-delta / current_temperature):
            current_solution = new_solution
            current_distance = new_distance

        # Update the best solution if the current solution is better
        if current_distance < best_distance:
            best_solution = current_solution
            best_distance = current_distance

    # Create the optimal route
    optimal_route = []
    start_index = best_solution.index(0)
    for i in range(num_locations):
        optimal_route.append(best_solution[(start_index+i)%num_locations])
    optimal_route.append(0)

    # Return the optimal route
    location_route = [locations[i] for i in optimal_route]
    return location_route

# Caching the distance matrix calculation for better performance
@st.cache_data
def compute_distance_matrix(locations):
    # using geopy geodesic for lesser compute time
    num_locations = len(locations)
    distance_matrix = [[0] * num_locations for i in range(num_locations)]
    for i in range(num_locations):
        for j in range(i, num_locations):
            distance = geodesic(locations[i], locations[j]).km
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
    return distance_matrix

def create_data_model(locations):
    data = {}
    num_locations = len(locations)
    data['locations'] = locations
    data['num_locations'] = num_locations
    distance_matrix = compute_distance_matrix(locations)
    data['distance_matrix'] = distance_matrix
    return data

def geocode_address(address):
    url = f'https://photon.komoot.io/api/?q={address}'
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json()
        if results['features']:
            first_result = results['features'][0]
            latitude = first_result['geometry']['coordinates'][1]
            longitude = first_result['geometry']['coordinates'][0]
            return address, latitude, longitude
        else:
            print(f'Geocode was not successful. No results found for address: {address}')
    else:
        print('Failed to get a response from the geocoding API.')

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

    # Display itinerary for each day
    for i, day in enumerate(days[1:num_days+1], start=1):
        day_itinerary = day.strip()

        st.subheader(f'Day {i} Itinerary:')
        st.write(day_itinerary)

        # Extract locations from the current day's itinerary
        day_locations = extract_locations(day_itinerary)

        # Geocode the locations
        geocoded_locations = [(loc, *geocode_address(loc)[1:]) for loc in day_locations]
        loc_df = pd.DataFrame(geocoded_locations, columns=['Place_Name', 'Latitude', 'Longitude'])

        # Create the data model for the TSP solver
        data_model = create_data_model([(row['Latitude'], row['Longitude']) for _, row in loc_df.iterrows()])

        # Solve the TSP problem and get the optimal route
        location_route = tsp_solver(data_model)

        # Generate and display Google Maps link with optimal route for the current day
        gmap_link = generate_google_maps_link(location_route, loc_df)
        st.write(f'[Google Maps Link for Day {i} Itinerary]({gmap_link})')

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