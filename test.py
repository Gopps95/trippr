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
Prepare a {num_days}-day trip schedule for {destination}, a vibrant city in the Indian state of Kerala, known for its rich cultural heritage, breathtaking backwaters, and diverse culinary delights. Here are the details:

* Arrival To: {arrival_to}
* Arrival Date: {arrival_date}
* Arrival Time: {arrival_time}

* Departure From: {departure_from}
* Departure Date: {departure_date}
* Departure Time: {departure_time}

* Additional Notes: {additional_information}

Include visits to popular attractions like Fort Kochi, Mattancherry Palace, Cherai Beach, and the backwaters of Kumarakom. Recommend opportunities to experience local cuisine, art forms like Kathakali, and shopping for spices and handicrafts.

Also, provide detailed daily route recommendations for exploring the destination while considering factors like traffic, weather, and distance between locations.
'''.strip()

def extract_locations(text):
    doc = nlp(text)
    locations = []
    for ent in doc.ents:
        if ent.label_ == "GPE" and ent.text.lower() not in ['kerala', 'india']:
            locations.append(ent.text)
    return locations

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

    # If there are less than 2 locations, return the original list
    if num_locations < 2:
        return locations

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
# Remove the second form definition
# Initialization
# Other imports and code...

# Initialization
if 'output' not in st.session_state:
    st.session_state['output'] = '--'

st.title('GPT-3 Trip Scheduler')
st.subheader('Let us plan your trip!')

# Initialization
if 'output' not in st.session_state:
    st.session_state['output'] = '--'

st.title('GPT-3 Trip Scheduler')
st.subheader('Let us plan your trip!')

# Define the form outside the columns
with st.form(key='trip_form'):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader('Destination')
        origin = st.text_input('Destination', value=random.choice(EXAMPLE_DESTINATIONS), key='destination')

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

    # Add a submit button
    submitted = st.form_submit_button("Generate Itinerary")

    if submitted:
        generate_itinerary()

# Display the output
st.subheader('Trip Schedule')
st.write(st.session_state.output)
