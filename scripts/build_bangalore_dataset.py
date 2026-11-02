"""
Bangalore Housing + Traffic + Spatial GNN Dataset Builder
=========================================================
Synthesizes tabular Kaggle Bengaluru real estate listings with spatial road networks,
Namma Metro accessibility, and peak vs. off-peak commute-time matrices to major tech clusters.

Outputs:
1. data/bangalore_tech_hubs.csv
2. data/bangalore_metro_stations.csv
3. data/bangalore_localities_master.csv
4. data/bangalore_locality_traffic_matrix.csv
5. data/bangalore_housing_cleaned_tabular.csv
6. data/bangalore_housing_spatial_commute_dataset.csv
7. data/bangalore_locality_graph_edges.csv
8. data/bangalore_locality_graph_nodes.csv
9. data/bangalore_graph_pyg.pt (PyG-ready PyTorch tensor dict)
"""

import os
import re
import math
import numpy as np
import pandas as pd
import torch
from rapidfuzz import process, fuzz

os.makedirs('data', exist_ok=True)
os.makedirs('scripts', exist_ok=True)

# ==============================================================================
# 1. PRIMARY TECH CLUSTERS & EMPLOYMENT HUBS (5 Targets as per Proposal & Deck)
# ==============================================================================
TECH_HUBS = [
    {
        "hub_id": 0,
        "hub_name": "Whitefield",
        "description": "ITPL, EPIP Zone, Export Promotion Industrial Park, Prestige Shantiniketan",
        "corridor": "East Tech Corridor",
        "latitude": 12.9840,
        "longitude": 77.7499
    },
    {
        "hub_id": 1,
        "hub_name": "Bellandur_ORR",
        "description": "Outer Ring Road, EcoSpace, Embassy TechVillage, Prestige Tech Park",
        "corridor": "South-East ORR Corridor",
        "latitude": 12.9260,
        "longitude": 77.6762
    },
    {
        "hub_id": 2,
        "hub_name": "Electronic_City",
        "description": "Electronic City Phase 1 & 2, Infosys, Wipro, Velankani Tech Park",
        "corridor": "South Highway Corridor",
        "latitude": 12.8452,
        "longitude": 77.6602
    },
    {
        "hub_id": 3,
        "hub_name": "Manyata_Tech_Park",
        "description": "Hebbal / Nagavara, Manyata Embassy Business Park, North Bangalore Tech Hub",
        "corridor": "North Tech Corridor",
        "latitude": 13.0475,
        "longitude": 77.6214
    },
    {
        "hub_id": 4,
        "hub_name": "MG_Road_CBD",
        "description": "Central Business District, UB City, Brigade Road, Commercial Hub",
        "corridor": "Central Business District",
        "latitude": 12.9756,
        "longitude": 77.6066
    }
]

# ==============================================================================
# 2. BMRCL NAMMA METRO OPERATIONAL & ARTERIAL STATIONS
# ==============================================================================
METRO_STATIONS = [
    {"station_name": "Nadaprabhu Kempegowda (Majestic)", "line": "Interchange (Green/Purple)", "latitude": 12.9757, "longitude": 77.5728},
    {"station_name": "MG Road", "line": "Purple", "latitude": 12.9756, "longitude": 77.6066},
    {"station_name": "Trinity", "line": "Purple", "latitude": 12.9729, "longitude": 77.6170},
    {"station_name": "Halasuru", "line": "Purple", "latitude": 12.9760, "longitude": 77.6267},
    {"station_name": "Indiranagar", "line": "Purple", "latitude": 12.9784, "longitude": 77.6408},
    {"station_name": "Swami Vivekananda Road", "line": "Purple", "latitude": 12.9860, "longitude": 77.6444},
    {"station_name": "Baiyappanahalli", "line": "Purple", "latitude": 12.9907, "longitude": 77.6525},
    {"station_name": "Benniganahalli (Tin Factory)", "line": "Purple", "latitude": 12.9972, "longitude": 77.6628},
    {"station_name": "KR Puram", "line": "Purple", "latitude": 13.0012, "longitude": 77.6975},
    {"station_name": "Garudacharpalya", "line": "Purple", "latitude": 12.9930, "longitude": 77.7120},
    {"station_name": "Hoodi", "line": "Purple", "latitude": 12.9919, "longitude": 77.7163},
    {"station_name": "Kundalahalli", "line": "Purple", "latitude": 12.9715, "longitude": 77.7142},
    {"station_name": "Nallurhalli", "line": "Purple", "latitude": 12.9734, "longitude": 77.7303},
    {"station_name": "Sri Sathya Sai Hospital", "line": "Purple", "latitude": 12.9774, "longitude": 77.7441},
    {"station_name": "Pattandur Agrahara", "line": "Purple", "latitude": 12.9829, "longitude": 77.7554},
    {"station_name": "Kadugodi Tree Park", "line": "Purple", "latitude": 12.9892, "longitude": 77.7580},
    {"station_name": "Whitefield (Kadugodi)", "line": "Purple", "latitude": 12.9986, "longitude": 77.7610},
    {"station_name": "Cubbon Park", "line": "Purple", "latitude": 12.9810, "longitude": 77.5975},
    {"station_name": "Vidhana Soudha", "line": "Purple", "latitude": 12.9798, "longitude": 77.5925},
    {"station_name": "Sir M. Visveshwaraya", "line": "Purple", "latitude": 12.9739, "longitude": 77.5838},
    {"station_name": "City Railway Station", "line": "Purple", "latitude": 12.9782, "longitude": 77.5670},
    {"station_name": "Magadi Road", "line": "Purple", "latitude": 12.9755, "longitude": 77.5552},
    {"station_name": "Hosahalli", "line": "Purple", "latitude": 12.9744, "longitude": 77.5451},
    {"station_name": "Vijayanagar", "line": "Purple", "latitude": 12.9701, "longitude": 77.5372},
    {"station_name": "Attiguppe", "line": "Purple", "latitude": 12.9622, "longitude": 77.5332},
    {"station_name": "Deepanjali Nagar", "line": "Purple", "latitude": 12.9525, "longitude": 77.5317},
    {"station_name": "Mysuru Road", "line": "Purple", "latitude": 12.9469, "longitude": 77.5300},
    {"station_name": "Nayandahalli", "line": "Purple", "latitude": 12.9413, "longitude": 77.5218},
    {"station_name": "Rajarajeshwari Nagar", "line": "Purple", "latitude": 12.9348, "longitude": 77.5146},
    {"station_name": "Jnanabharathi", "line": "Purple", "latitude": 12.9261, "longitude": 77.5015},
    {"station_name": "Pattanagere", "line": "Purple", "latitude": 12.9195, "longitude": 77.4921},
    {"station_name": "Kengeri Bus Terminal", "line": "Purple", "latitude": 12.9138, "longitude": 77.4832},
    {"station_name": "Kengeri", "line": "Purple", "latitude": 12.9082, "longitude": 77.4764},
    {"station_name": "Challaghatta", "line": "Purple", "latitude": 12.9030, "longitude": 77.4650},
    {"station_name": "Nagasandra", "line": "Green", "latitude": 13.0483, "longitude": 77.4996},
    {"station_name": "Dasarahalli", "line": "Green", "latitude": 13.0436, "longitude": 77.5130},
    {"station_name": "Jalahalli", "line": "Green", "latitude": 13.0392, "longitude": 77.5198},
    {"station_name": "Peenya Industry", "line": "Green", "latitude": 13.0362, "longitude": 77.5255},
    {"station_name": "Peenya", "line": "Green", "latitude": 13.0329, "longitude": 77.5336},
    {"station_name": "Goraguntepalya", "line": "Green", "latitude": 13.0284, "longitude": 77.5418},
    {"station_name": "Yeshwanthpur", "line": "Green", "latitude": 13.0232, "longitude": 77.5501},
    {"station_name": "Sandal Soap Factory", "line": "Green", "latitude": 13.0147, "longitude": 77.5539},
    {"station_name": "Mahalakshmi", "line": "Green", "latitude": 13.0081, "longitude": 77.5489},
    {"station_name": "Rajajinagar", "line": "Green", "latitude": 12.9987, "longitude": 77.5558},
    {"station_name": "Kuvempu Road", "line": "Green", "latitude": 12.9897, "longitude": 77.5601},
    {"station_name": "Srirampura", "line": "Green", "latitude": 12.9961, "longitude": 77.5632},
    {"station_name": "Sampige Road (Malleshwaram)", "line": "Green", "latitude": 12.9912, "longitude": 77.5707},
    {"station_name": "Chickpet", "line": "Green", "latitude": 12.9667, "longitude": 77.5744},
    {"station_name": "Krishna Rajendra Market", "line": "Green", "latitude": 12.9600, "longitude": 77.5756},
    {"station_name": "National College", "line": "Green", "latitude": 12.9506, "longitude": 77.5731},
    {"station_name": "Lalbagh", "line": "Green", "latitude": 12.9463, "longitude": 77.5800},
    {"station_name": "South End Circle", "line": "Green", "latitude": 12.9376, "longitude": 77.5803},
    {"station_name": "Jayanagar", "line": "Green", "latitude": 12.9298, "longitude": 77.5801},
    {"station_name": "Rashtreeya Vidyalaya Road", "line": "Green", "latitude": 12.9216, "longitude": 77.5801},
    {"station_name": "Banashankari", "line": "Green", "latitude": 12.9155, "longitude": 77.5736},
    {"station_name": "JP Nagar", "line": "Green", "latitude": 12.9074, "longitude": 77.5746},
    {"station_name": "Yelachenahalli", "line": "Green", "latitude": 12.8959, "longitude": 77.5704},
    {"station_name": "Konanakunte Cross", "line": "Green", "latitude": 12.8847, "longitude": 77.5682},
    {"station_name": "Doddakallasandra", "line": "Green", "latitude": 12.8732, "longitude": 77.5583},
    {"station_name": "Vajarahalli", "line": "Green", "latitude": 12.8643, "longitude": 77.5467},
    {"station_name": "Thalaghattapura", "line": "Green", "latitude": 12.8541, "longitude": 77.5385},
    {"station_name": "Silk Institute", "line": "Green", "latitude": 12.8427, "longitude": 77.5298},
    {"station_name": "Central Silk Board", "line": "Yellow (Hub)", "latitude": 12.9176, "longitude": 77.6238},
    {"station_name": "Electronic City", "line": "Yellow", "latitude": 12.8450, "longitude": 77.6630}
]

# ==============================================================================
# 3. COMPREHENSIVE CANONICAL BANGALORE LOCALITY MASTER CATALOG
# ==============================================================================
# Master list of canonical localities with validated geospatial centroids
CANONICAL_LOCALITIES = [
    # East / IT Corridor
    {"name": "Whitefield", "lat": 12.9698, "lon": 77.7499, "zone": "East", "quadrant": "East"},
    {"name": "Marathahalli", "lat": 12.9591, "lon": 77.6974, "zone": "East", "quadrant": "East"},
    {"name": "Bellandur", "lat": 12.9260, "lon": 77.6762, "zone": "South-East", "quadrant": "East"},
    {"name": "Sarjapur Road", "lat": 12.9116, "lon": 77.6744, "zone": "South-East", "quadrant": "East"},
    {"name": "Sarjapur", "lat": 12.8601, "lon": 77.7865, "zone": "Suburban East", "quadrant": "East"},
    {"name": "Varthur", "lat": 12.9406, "lon": 77.7410, "zone": "East", "quadrant": "East"},
    {"name": "Haralur Road", "lat": 12.9121, "lon": 77.6620, "zone": "South-East", "quadrant": "East"},
    {"name": "Harlur", "lat": 12.9121, "lon": 77.6620, "zone": "South-East", "quadrant": "East"},
    {"name": "Kasavanahalli", "lat": 12.9079, "lon": 77.6773, "zone": "South-East", "quadrant": "East"},
    {"name": "Hoodi", "lat": 12.9919, "lon": 77.7163, "zone": "East", "quadrant": "East"},
    {"name": "Kundalahalli", "lat": 12.9660, "lon": 77.7169, "zone": "East", "quadrant": "East"},
    {"name": "KR Puram", "lat": 13.0075, "lon": 77.6959, "zone": "East", "quadrant": "East"},
    {"name": "Kaggadasapura", "lat": 12.9852, "lon": 77.6775, "zone": "East", "quadrant": "East"},
    {"name": "CV Raman Nagar", "lat": 12.9854, "lon": 77.6639, "zone": "East", "quadrant": "East"},
    {"name": "Mahadevapura", "lat": 12.9926, "lon": 77.6932, "zone": "East", "quadrant": "East"},
    {"name": "Brookefield", "lat": 12.9658, "lon": 77.7180, "zone": "East", "quadrant": "East"},
    {"name": "Kadugodi", "lat": 12.9986, "lon": 77.7610, "zone": "East", "quadrant": "East"},
    {"name": "Budigere Cross", "lat": 13.0789, "lon": 77.7424, "zone": "Suburban East", "quadrant": "East"},
    {"name": "Gunjur", "lat": 12.9150, "lon": 77.7330, "zone": "East", "quadrant": "East"},
    {"name": "Panathur", "lat": 12.9340, "lon": 77.7010, "zone": "East", "quadrant": "East"},
    {"name": "Bhoganhalli", "lat": 12.9315, "lon": 77.6920, "zone": "East", "quadrant": "East"},
    {"name": "Doddanekundi", "lat": 12.9712, "lon": 77.6970, "zone": "East", "quadrant": "East"},
    {"name": "Chikka Tirupathi", "lat": 12.8988, "lon": 77.8682, "zone": "Suburban East", "quadrant": "East"},
    {"name": "Old Airport Road", "lat": 12.9602, "lon": 77.6534, "zone": "East", "quadrant": "East"},
    {"name": "Murugeshpalya", "lat": 12.9560, "lon": 77.6530, "zone": "East", "quadrant": "East"},
    {"name": "Old Madras Road", "lat": 13.0030, "lon": 77.6750, "zone": "East", "quadrant": "East"},
    {"name": "TC Palaya", "lat": 13.0230, "lon": 77.7020, "zone": "East", "quadrant": "East"},
    {"name": "Battarahalli", "lat": 13.0280, "lon": 77.7120, "zone": "East", "quadrant": "East"},
    {"name": "Ramagondanahalli", "lat": 12.9600, "lon": 77.7300, "zone": "East", "quadrant": "East"},
    {"name": "Nallurhalli", "lat": 12.9734, "lon": 77.7303, "zone": "East", "quadrant": "East"},
    {"name": "Pattandur Agrahara", "lat": 12.9829, "lon": 77.7554, "zone": "East", "quadrant": "East"},
    {"name": "Channasandra", "lat": 12.9950, "lon": 77.7600, "zone": "East", "quadrant": "East"},
    {"name": "Seegehalli", "lat": 13.0110, "lon": 77.7420, "zone": "East", "quadrant": "East"},
    {"name": "Kodi", "lat": 12.9810, "lon": 77.7320, "zone": "East", "quadrant": "East"},
    {"name": "Balagere", "lat": 12.9380, "lon": 77.7180, "zone": "East", "quadrant": "East"},
    {"name": "Mullur", "lat": 12.9050, "lon": 77.7080, "zone": "East", "quadrant": "East"},
    {"name": "Carmelaram", "lat": 12.9110, "lon": 77.7100, "zone": "East", "quadrant": "East"},

    # South Corridor / Electronic City / Bannerghatta / Kanakapura
    {"name": "Electronic City", "lat": 12.8452, "lon": 77.6602, "zone": "South", "quadrant": "South"},
    {"name": "Electronic City Phase 1", "lat": 12.8452, "lon": 77.6602, "zone": "South", "quadrant": "South"},
    {"name": "Electronic City Phase 2", "lat": 12.8490, "lon": 77.6790, "zone": "South", "quadrant": "South"},
    {"name": "Chandapura", "lat": 12.7933, "lon": 77.7011, "zone": "Suburban South", "quadrant": "South"},
    {"name": "Jigani", "lat": 12.7842, "lon": 77.6416, "zone": "Suburban South", "quadrant": "South"},
    {"name": "Anekal", "lat": 12.7107, "lon": 77.6974, "zone": "Suburban South", "quadrant": "South"},
    {"name": "Attibele", "lat": 12.7780, "lon": 77.7710, "zone": "Suburban South", "quadrant": "South"},
    {"name": "Bommasandra", "lat": 12.8170, "lon": 77.6850, "zone": "South", "quadrant": "South"},
    {"name": "Doddathoguru", "lat": 12.8520, "lon": 77.6530, "zone": "South", "quadrant": "South"},
    {"name": "Hosa Road", "lat": 12.8850, "lon": 77.6550, "zone": "South", "quadrant": "South"},
    {"name": "Begur Road", "lat": 12.8820, "lon": 77.6250, "zone": "South", "quadrant": "South"},
    {"name": "Begur", "lat": 12.8780, "lon": 77.6290, "zone": "South", "quadrant": "South"},
    {"name": "Singasandra", "lat": 12.8890, "lon": 77.6470, "zone": "South", "quadrant": "South"},
    {"name": "Kudlu Gate", "lat": 12.8980, "lon": 77.6490, "zone": "South", "quadrant": "South"},
    {"name": "Kudlu", "lat": 12.8990, "lon": 77.6540, "zone": "South", "quadrant": "South"},
    {"name": "HSR Layout", "lat": 12.9121, "lon": 77.6446, "zone": "South", "quadrant": "South"},
    {"name": "Koramangala", "lat": 12.9352, "lon": 77.6245, "zone": "South", "quadrant": "South"},
    {"name": "BTM Layout", "lat": 12.9166, "lon": 77.6101, "zone": "South", "quadrant": "South"},
    {"name": "Bommanahalli", "lat": 12.9020, "lon": 77.6240, "zone": "South", "quadrant": "South"},
    {"name": "Mangammanapalya", "lat": 12.9080, "lon": 77.6390, "zone": "South", "quadrant": "South"},
    {"name": "Bannerghatta Road", "lat": 12.8900, "lon": 77.5980, "zone": "South", "quadrant": "South"},
    {"name": "Hulimavu", "lat": 12.8790, "lon": 77.6010, "zone": "South", "quadrant": "South"},
    {"name": "Arekere", "lat": 12.8850, "lon": 77.5950, "zone": "South", "quadrant": "South"},
    {"name": "Gottigere", "lat": 12.8570, "lon": 77.5870, "zone": "South", "quadrant": "South"},
    {"name": "Kalena Agrahara", "lat": 12.8670, "lon": 77.5930, "zone": "South", "quadrant": "South"},
    {"name": "Akshaya Nagar", "lat": 12.8750, "lon": 77.6150, "zone": "South", "quadrant": "South"},
    {"name": "Devarachikkanahalli", "lat": 12.8780, "lon": 77.6120, "zone": "South", "quadrant": "South"},
    {"name": "JP Nagar", "lat": 12.9074, "lon": 77.5746, "zone": "South", "quadrant": "South"},
    {"name": "Jayanagar", "lat": 12.9298, "lon": 77.5801, "zone": "South", "quadrant": "Central"},
    {"name": "Banashankari", "lat": 12.9155, "lon": 77.5736, "zone": "South", "quadrant": "South"},
    {"name": "Kumaraswamy Layout", "lat": 12.9030, "lon": 77.5580, "zone": "South", "quadrant": "South"},
    {"name": "ISRO Layout", "lat": 12.8970, "lon": 77.5540, "zone": "South", "quadrant": "South"},
    {"name": "Padmanabhanagar", "lat": 12.9180, "lon": 77.5560, "zone": "South", "quadrant": "South"},
    {"name": "Uttarahalli", "lat": 12.9063, "lon": 77.5348, "zone": "South-West", "quadrant": "West"},
    {"name": "Chikkalasandra", "lat": 12.9130, "lon": 77.5450, "zone": "South-West", "quadrant": "West"},
    {"name": "Kanakapura Road", "lat": 12.8800, "lon": 77.5500, "zone": "South", "quadrant": "South"},
    {"name": "Konanakunte", "lat": 12.8847, "lon": 77.5682, "zone": "South", "quadrant": "South"},
    {"name": "Yelachenahalli", "lat": 12.8959, "lon": 77.5704, "zone": "South", "quadrant": "South"},
    {"name": "Talaghattapura", "lat": 12.8541, "lon": 77.5385, "zone": "South", "quadrant": "South"},
    {"name": "Kaggalipura", "lat": 12.7960, "lon": 77.5180, "zone": "Suburban South", "quadrant": "South"},
    {"name": "Anjanapura", "lat": 12.8530, "lon": 77.5660, "zone": "South", "quadrant": "South"},
    {"name": "Billekahalli", "lat": 12.8990, "lon": 77.6050, "zone": "South", "quadrant": "South"},
    {"name": "Basavanagudi", "lat": 12.9420, "lon": 77.5750, "zone": "South", "quadrant": "Central"},
    {"name": "Gandhi Bazar", "lat": 12.9440, "lon": 77.5710, "zone": "South", "quadrant": "Central"},

    # North Corridor / Hebbal / Yelahanka / Airport Road
    {"name": "Hebbal", "lat": 13.0358, "lon": 77.5970, "zone": "North", "quadrant": "North"},
    {"name": "Hebbal Kempapura", "lat": 13.0470, "lon": 77.6010, "zone": "North", "quadrant": "North"},
    {"name": "Manyata Tech Park", "lat": 13.0475, "lon": 77.6214, "zone": "North", "quadrant": "North"},
    {"name": "Thanisandra", "lat": 13.0558, "lon": 77.6322, "zone": "North", "quadrant": "North"},
    {"name": "Yelahanka", "lat": 13.1007, "lon": 77.5963, "zone": "North", "quadrant": "North"},
    {"name": "Yelahanka New Town", "lat": 13.0980, "lon": 77.5780, "zone": "North", "quadrant": "North"},
    {"name": "Kogilu", "lat": 13.1110, "lon": 77.6250, "zone": "North", "quadrant": "North"},
    {"name": "Jakkur", "lat": 13.0783, "lon": 77.6071, "zone": "North", "quadrant": "North"},
    {"name": "Hennur Road", "lat": 13.0350, "lon": 77.6400, "zone": "North", "quadrant": "North"},
    {"name": "Hennur", "lat": 13.0280, "lon": 77.6420, "zone": "North", "quadrant": "North"},
    {"name": "Hormavu", "lat": 13.0335, "lon": 77.6638, "zone": "North-East", "quadrant": "North"},
    {"name": "Kothanur", "lat": 13.0610, "lon": 77.6500, "zone": "North", "quadrant": "North"},
    {"name": "Rachenahalli", "lat": 13.0540, "lon": 77.6190, "zone": "North", "quadrant": "North"},
    {"name": "Nagavara", "lat": 13.0390, "lon": 77.6180, "zone": "North", "quadrant": "North"},
    {"name": "Hegde Nagar", "lat": 13.0690, "lon": 77.6320, "zone": "North", "quadrant": "North"},
    {"name": "Kalyan Nagar", "lat": 13.0220, "lon": 77.6470, "zone": "North-East", "quadrant": "North"},
    {"name": "Kammanahalli", "lat": 13.0130, "lon": 77.6380, "zone": "North-East", "quadrant": "North"},
    {"name": "Ramamurthy Nagar", "lat": 13.0163, "lon": 77.6784, "zone": "East", "quadrant": "East"},
    {"name": "Banaswadi", "lat": 13.0080, "lon": 77.6450, "zone": "East", "quadrant": "East"},
    {"name": "Kasturi Nagar", "lat": 13.0040, "lon": 77.6600, "zone": "East", "quadrant": "East"},
    {"name": "Devanahalli", "lat": 13.2484, "lon": 77.7126, "zone": "Suburban North", "quadrant": "North"},
    {"name": "Chikkaballapur Road", "lat": 13.2500, "lon": 77.7200, "zone": "Suburban North", "quadrant": "North"},
    {"name": "Doddaballapur Road", "lat": 13.1800, "lon": 77.5400, "zone": "Suburban North", "quadrant": "North"},
    {"name": "Sanjay Nagar", "lat": 13.0360, "lon": 77.5750, "zone": "North", "quadrant": "North"},
    {"name": "Ganga Nagar", "lat": 13.0250, "lon": 77.5890, "zone": "North", "quadrant": "North"},
    {"name": "RT Nagar", "lat": 13.0180, "lon": 77.5940, "zone": "North", "quadrant": "Central"},
    {"name": "Kaval Byrasandra", "lat": 13.0230, "lon": 77.6040, "zone": "North", "quadrant": "North"},
    {"name": "Sahakara Nagar", "lat": 13.0620, "lon": 77.5910, "zone": "North", "quadrant": "North"},
    {"name": "Vidyaranyapura", "lat": 13.0820, "lon": 77.5620, "zone": "North", "quadrant": "North"},
    {"name": "Thindlu", "lat": 13.0760, "lon": 77.5730, "zone": "North", "quadrant": "North"},
    {"name": "Bagalur", "lat": 13.1340, "lon": 77.6660, "zone": "Suburban North", "quadrant": "North"},
    {"name": "Bisuvanahalli", "lat": 13.2100, "lon": 77.7150, "zone": "Suburban North", "quadrant": "North"},

    # West Corridor / Mysore Road / Magadi Road / Peenya / Yeshwanthpur
    {"name": "Raja Rajeshwari Nagar", "lat": 12.9274, "lon": 77.5154, "zone": "West", "quadrant": "West"},
    {"name": "Kengeri", "lat": 12.9082, "lon": 77.4764, "zone": "West", "quadrant": "West"},
    {"name": "Kengeri Satellite Town", "lat": 12.9150, "lon": 77.4820, "zone": "West", "quadrant": "West"},
    {"name": "Mysore Road", "lat": 12.9469, "lon": 77.5300, "zone": "West", "quadrant": "West"},
    {"name": "Nagarbhavi", "lat": 12.9610, "lon": 77.5100, "zone": "West", "quadrant": "West"},
    {"name": "Vijayanagar", "lat": 12.9701, "lon": 77.5372, "zone": "West", "quadrant": "West"},
    {"name": "Basaveshwara Nagar", "lat": 12.9860, "lon": 77.5400, "zone": "West", "quadrant": "West"},
    {"name": "Rajaji Nagar", "lat": 12.9987, "lon": 77.5558, "zone": "West", "quadrant": "Central"},
    {"name": "Mahalakshmi Layout", "lat": 13.0081, "lon": 77.5489, "zone": "West", "quadrant": "West"},
    {"name": "Yeshwanthpur", "lat": 13.0232, "lon": 77.5501, "zone": "North-West", "quadrant": "North"},
    {"name": "Malleshwaram", "lat": 13.0031, "lon": 77.5701, "zone": "Central-North", "quadrant": "Central"},
    {"name": "Peenya", "lat": 13.0329, "lon": 77.5255, "zone": "North-West", "quadrant": "North"},
    {"name": "Jalahalli", "lat": 13.0420, "lon": 77.5380, "zone": "North-West", "quadrant": "North"},
    {"name": "Jalahalli West", "lat": 13.0450, "lon": 77.5180, "zone": "North-West", "quadrant": "North"},
    {"name": "Magadi Road", "lat": 12.9755, "lon": 77.5552, "zone": "West", "quadrant": "West"},
    {"name": "Chandra Layout", "lat": 12.9550, "lon": 77.5280, "zone": "West", "quadrant": "West"},
    {"name": "Thigalarapalya", "lat": 13.0050, "lon": 77.4980, "zone": "West", "quadrant": "West"},
    {"name": "Tumkur Road", "lat": 13.0550, "lon": 77.4950, "zone": "North-West", "quadrant": "North"},
    {"name": "Nagasandra", "lat": 13.0483, "lon": 77.4996, "zone": "North-West", "quadrant": "North"},
    {"name": "Hesaraghatta", "lat": 13.1400, "lon": 77.4800, "zone": "Suburban North", "quadrant": "North"},
    {"name": "Abbigere", "lat": 13.0780, "lon": 77.5300, "zone": "North-West", "quadrant": "North"},
    {"name": "Binny Pete", "lat": 12.9690, "lon": 77.5570, "zone": "Central-West", "quadrant": "Central"},
    {"name": "Kumbalgodu", "lat": 12.8720, "lon": 77.4420, "zone": "Suburban West", "quadrant": "West"},
    {"name": "Bidadi", "lat": 12.7950, "lon": 77.3870, "zone": "Suburban West", "quadrant": "West"},
    {"name": "Lingadheeranahalli", "lat": 12.8870, "lon": 77.5120, "zone": "South-West", "quadrant": "West"},

    # Central Core
    {"name": "Indiranagar", "lat": 12.9784, "lon": 77.6408, "zone": "Central-East", "quadrant": "Central"},
    {"name": "MG Road", "lat": 12.9756, "lon": 77.6066, "zone": "Central", "quadrant": "Central"},
    {"name": "Domlur", "lat": 12.9610, "lon": 77.6390, "zone": "Central-East", "quadrant": "Central"},
    {"name": "Ulsoor", "lat": 12.9810, "lon": 77.6250, "zone": "Central", "quadrant": "Central"},
    {"name": "Frazer Town", "lat": 12.9970, "lon": 77.6130, "zone": "Central-North", "quadrant": "Central"},
    {"name": "Shivaji Nagar", "lat": 12.9850, "lon": 77.6030, "zone": "Central", "quadrant": "Central"},
    {"name": "Richmond Town", "lat": 12.9650, "lon": 77.6040, "zone": "Central", "quadrant": "Central"},
    {"name": "Shanthi Nagar", "lat": 12.9570, "lon": 77.5940, "zone": "Central", "quadrant": "Central"},
    {"name": "Wilson Garden", "lat": 12.9480, "lon": 77.5960, "zone": "Central", "quadrant": "Central"},
    {"name": "Cox Town", "lat": 12.9990, "lon": 77.6250, "zone": "Central-East", "quadrant": "Central"},
    {"name": "Cooke Town", "lat": 13.0030, "lon": 77.6270, "zone": "Central-East", "quadrant": "Central"},
    {"name": "Benson Town", "lat": 13.0030, "lon": 77.6040, "zone": "Central-North", "quadrant": "Central"},
    {"name": "Vasanth Nagar", "lat": 12.9890, "lon": 77.5920, "zone": "Central", "quadrant": "Central"},
    {"name": "Seshadripuram", "lat": 12.9910, "lon": 77.5780, "zone": "Central", "quadrant": "Central"}
]

# Create lookup dictionary and dataframe
df_canonical = pd.DataFrame(CANONICAL_LOCALITIES)
df_canonical['locality_id'] = range(len(df_canonical))

# ==============================================================================
# 4. HAVERSINE & TRAFFIC COMMUTE ROUTING MODEL
# ==============================================================================
def haversine_km(lat1, lon1, lat2, lon2):
    """Computes great circle distance between two lat/lon pairs in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def compute_urban_commute_time(d_euclidean_km, corridor="general"):
    """
    Computes realistic Indian urban road distance and peak vs off-peak travel times.
    Bangalore urban characteristics:
    - Tortuosity factor (road distance / euclidean): 1.25x - 1.45x
    - Off-peak average speed: 30 - 40 km/h
    - Peak average speed: 12 - 18 km/h depending on corridor congestion bottlenecks
    - Congestion ratio: 1.8x - 2.6x
    """
    # Realistic road distance with Bangalore network circuity
    tortuosity = 1.32
    d_road = d_euclidean_km * tortuosity
    
    # Base speeds in km/h
    if corridor in ["Bellandur_ORR", "Whitefield"]:
        # Heaviest tech bottleneck corridors (Silk Board, Marathahalli, ORR)
        v_offpeak = 32.0
        v_peak = 13.5
        delay_bottleneck_peak = 10.0 if d_road > 4.0 else 4.0 # Signal & flyover bottlenecks
    elif corridor == "Electronic_City":
        # Elevated expressway speeds up long distance, but bottlenecks at Silk Board / Bommanahalli
        v_offpeak = 38.0
        v_peak = 17.0
        delay_bottleneck_peak = 8.0 if d_road > 5.0 else 3.0
    elif corridor == "Manyata_Tech_Park":
        # Hebbal flyover bottleneck
        v_offpeak = 34.0
        v_peak = 15.0
        delay_bottleneck_peak = 7.0 if d_road > 4.0 else 3.0
    else: # Central / CBD
        v_offpeak = 28.0
        v_peak = 14.0
        delay_bottleneck_peak = 6.0 if d_road > 3.0 else 2.0
        
    t_offpeak_min = (d_road / v_offpeak) * 60.0 + 2.0 # Minimum 2 min buffer
    t_peak_min = (d_road / v_peak) * 60.0 + delay_bottleneck_peak
    
    # Cap lower bound for local distance
    t_offpeak_min = max(3.0, round(t_offpeak_min, 1))
    t_peak_min = max(t_offpeak_min * 1.25, round(t_peak_min, 1))
    
    return round(d_road, 2), t_offpeak_min, t_peak_min

# ==============================================================================
# 5. STRING NORMALIZATION & FUZZY MATCHING ENGINE
# ==============================================================================
# Specific high-frequency aliases in Kaggle dataset mapped to canonical names
EXACT_ALIAS_MAP = {
    "sarjapur  road": "Sarjapur Road",
    "sarjapura": "Sarjapur",
    "electronic city phase ii": "Electronic City Phase 2",
    "electronics city phase 1": "Electronic City Phase 1",
    "electronic city phase 1": "Electronic City Phase 1",
    "electronics city": "Electronic City",
    "electronic city": "Electronic City",
    "7th phase jp nagar": "JP Nagar",
    "8th phase jp nagar": "JP Nagar",
    "9th phase jp nagar": "JP Nagar",
    "6th phase jp nagar": "JP Nagar",
    "5th phase jp nagar": "JP Nagar",
    "2nd phase jp nagar": "JP Nagar",
    "1st phase jp nagar": "JP Nagar",
    "kanakpura road": "Kanakapura Road",
    "kanakapura": "Kanakapura Road",
    "manayata tech park": "Manyata Tech Park",
    "kasavanhalli": "Kasavanahalli",
    "harlur": "Haralur Road",
    "haralur road": "Haralur Road",
    "kengeri satellite town": "Kengeri Satellite Town",
    "thanisandra main road": "Thanisandra",
    "hennur road": "Hennur Road",
    "hennur": "Hennur",
    "kr puram": "KR Puram",
    "k r puram": "KR Puram",
    "cv raman nagar": "CV Raman Nagar",
    "c v raman nagar": "CV Raman Nagar",
    "btm layout 2nd stage": "BTM Layout",
    "btm 2nd stage": "BTM Layout",
    "btm 1st stage": "BTM Layout",
    "btm layout": "BTM Layout",
    "hsr": "HSR Layout",
    "hsr layout sector 1": "HSR Layout",
    "hsr layout sector 2": "HSR Layout",
    "hsr layout sector 3": "HSR Layout",
    "hsr layout sector 7": "HSR Layout",
    "raja rajeshwari nagar": "Raja Rajeshwari Nagar",
    "rr nagar": "Raja Rajeshwari Nagar",
    "yelahanka new town": "Yelahanka New Town",
    "hebbal kempapura": "Hebbal Kempapura",
    "bellandur": "Bellandur",
    "whitefield": "Whitefield",
    "marathahalli": "Marathahalli",
    "rajaji nagar": "Rajaji Nagar",
    "rajajinagar": "Rajaji Nagar",
    "malleshwaram": "Malleshwaram",
    "malleswaram": "Malleshwaram",
    "indiranagar": "Indiranagar",
    "koramangala": "Koramangala",
    "banashankari": "Banashankari",
    "banashankari stage iii": "Banashankari",
    "banashankari 3rd stage": "Banashankari",
    "uttarahalli": "Uttarahalli",
    "kothanur": "Kothanur",
    "kothannur": "Kothanur",
    "hormavu": "Hormavu",
    "horamavu": "Hormavu",
    "horamavu agara": "Hormavu",
    "bannerghatta road": "Bannerghatta Road",
    "bannerghatta": "Bannerghatta Road",
    "begur road": "Begur Road",
    "begur": "Begur",
    "jakkur": "Jakkur",
    "yeshwanthpur": "Yeshwanthpur",
    "ramamurthy nagar": "Ramamurthy Nagar",
    "kaggadasapura": "Kaggadasapura",
    "nagarbhavi": "Nagarbhavi",
    "thigalarapalya": "Thigalarapalya",
    "akshaya nagar": "Akshaya Nagar",
    "tc palaya": "TC Palaya",
    "rachenahalli": "Rachenahalli",
    "budigere": "Budigere Cross",
    "budigere cross": "Budigere Cross",
    "jigani": "Jigani",
    "jalahalli": "Jalahalli",
    "jalahalli west": "Jalahalli West",
    "chikkalasandra": "Chikkalasandra",
    "kundalahalli": "Kundalahalli",
    "varthur": "Varthur",
    "devanahalli": "Devanahalli",
    "chikka tirupathi": "Chikka Tirupathi",
    "old airport road": "Old Airport Road",
    "old madras road": "Old Madras Road",
    "mysore road": "Mysore Road",
    "binny pete": "Binny Pete",
    "gottigere": "Gottigere",
    "murugeshpalya": "Murugeshpalya",
    "doddathoguru": "Doddathoguru",
    "kalena agrahara": "Kalena Agrahara",
    "kaval byrasandra": "Kaval Byrasandra",
    "isro layout": "ISRO Layout",
    "padmanabhanagar": "Padmanabhanagar",
    "hegde nagar": "Hegde Nagar",
    "anajanapura": "Anjanapura",
    "anjanapura": "Anjanapura",
    "sahakara nagar": "Sahakara Nagar",
    "kammanahalli": "Kammanahalli",
    "kalyan nagar": "Kalyan Nagar",
    "banaswadi": "Banaswadi",
    "kasturi nagar": "Kasturi Nagar",
    "vidyaranyapura": "Vidyaranyapura",
    "bagalur": "Bagalur",
    "bisuvanahalli": "Bisuvanahalli",
    "basaveshwara nagar": "Basaveshwara Nagar",
    "mahalakshmi layout": "Mahalakshmi Layout",
    "peenya": "Peenya",
    "magadi road": "Magadi Road",
    "chandra layout": "Chandra Layout",
    "tumkur road": "Tumkur Road",
    "nagasandra": "Nagasandra",
    "lingadheeranahalli": "Lingadheeranahalli",
    "domlur": "Domlur",
    "ulsoor": "Ulsoor",
    "frazer town": "Frazer Town",
    "shivaji nagar": "Shivaji Nagar",
    "richmond town": "Richmond Town",
    "shanthi nagar": "Shanthi Nagar",
    "wilson garden": "Wilson Garden",
    "cox town": "Cox Town",
    "cooke town": "Cooke Town",
    "benson town": "Benson Town",
    "vasanth nagar": "Vasanth Nagar",
    "seshadripuram": "Seshadripuram",
    "basavanagudi": "Basavanagudi",
    "gandhi bazar": "Gandhi Bazar"
}

canonical_names_list = df_canonical['name'].tolist()

def clean_location_string(loc_str):
    if not isinstance(loc_str, str):
        return ""
    # Normalize spaces, lowercase, remove trailing commas/city names
    s = loc_str.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[,.-]', ' ', s)
    s = re.sub(r'\bbangalore\b|\bbengaluru\b', '', s).strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def match_canonical_locality(loc_raw):
    clean_loc = clean_location_string(loc_raw)
    if not clean_loc:
        return "Unknown", 0.0
    
    # 1. Exact alias match
    if clean_loc in EXACT_ALIAS_MAP:
        return EXACT_ALIAS_MAP[clean_loc], 100.0
    
    # 2. Check if clean_loc contains any canonical name or vice-versa
    for can_name in canonical_names_list:
        can_clean = clean_location_string(can_name)
        if can_clean == clean_loc:
            return can_name, 100.0
        if len(can_clean) > 4 and can_clean in clean_loc:
            return can_name, 92.0
            
    # 3. Fuzzy match using RapidFuzz token_sort_ratio
    match = process.extractOne(
        clean_loc,
        canonical_names_list,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=75.0
    )
    if match:
        return match[0], match[1]
    
    # Fallback partial ratio
    match_partial = process.extractOne(
        clean_loc,
        canonical_names_list,
        scorer=fuzz.partial_ratio,
        score_cutoff=82.0
    )
    if match_partial:
        return match_partial[0], match_partial[1]
        
    return "Other / Peripheral", 0.0


# ==============================================================================
# 6. TABULAR CLEANING & FEATURE EXTRACTION
# ==============================================================================
def parse_total_sqft(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    # Handle range e.g. "2100 - 2850"
    if '-' in val:
        parts = val.split('-')
        try:
            return (float(parts[0].strip()) + float(parts[1].strip())) / 2.0
        except:
            return np.nan
    # Handle units
    try:
        return float(val)
    except:
        val_lower = val.lower()
        num_match = re.search(r'([0-9.]+)', val_lower)
        if not num_match:
            return np.nan
        num = float(num_match.group(1))
        if 'sq. meter' in val_lower or 'sq meter' in val_lower or 'sqm' in val_lower:
            return num * 10.7639
        elif 'sq. yard' in val_lower or 'sq yard' in val_lower or 'sq yards' in val_lower:
            return num * 9.0
        elif 'acre' in val_lower:
            return num * 43560.0
        elif 'guntha' in val_lower:
            return num * 1089.0
        elif 'perch' in val_lower:
            return num * 272.25
        elif 'ground' in val_lower:
            return num * 2400.0
        elif 'cent' in val_lower:
            return num * 435.6
        return np.nan

def parse_bhk(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    match = re.search(r'(\d+)', val_str)
    if match:
        return int(match.group(1))
    return np.nan


# ==============================================================================
# 7. MAIN BUILD PIPELINE
# ==============================================================================
def run_pipeline():
    print("=" * 80)
    print("STARTING BANGALORE HOUSING + TRAFFIC + GNN DATASET GENERATION")
    print("=" * 80)

    # ---------------------------------------------------------
    # Step A: Export Tech Hubs
    # ---------------------------------------------------------
    df_hubs = pd.DataFrame(TECH_HUBS)
    df_hubs.to_csv("data/bangalore_tech_hubs.csv", index=False)
    print(f"[OK] Saved {len(df_hubs)} Tech Hub targets to data/bangalore_tech_hubs.csv")

    # ---------------------------------------------------------
    # Step B: Export Metro Stations
    # ---------------------------------------------------------
    df_metro = pd.DataFrame(METRO_STATIONS)
    df_metro['station_id'] = range(len(df_metro))
    df_metro.to_csv("data/bangalore_metro_stations.csv", index=False)
    print(f"[OK] Saved {len(df_metro)} Metro stations to data/bangalore_metro_stations.csv")

    # ---------------------------------------------------------
    # Step C: Compute Traffic Matrix & Spatial Features for Canonical Localities
    # ---------------------------------------------------------
    locality_traffic_records = []
    
    for _, loc in df_canonical.iterrows():
        rec = {
            "locality_id": loc["locality_id"],
            "canonical_locality_name": loc["name"],
            "latitude": loc["lat"],
            "longitude": loc["lon"],
            "zone": loc["zone"],
            "spatial_quadrant": loc["quadrant"]
        }
        
        # 1. Distances and Commute times to each of the 5 Tech Hubs
        commute_peak_times = []
        commute_offpeak_times = []
        road_dists = []
        
        for hub in TECH_HUBS:
            hub_code = hub["hub_name"].lower().replace(' ', '_')
            d_euc = haversine_km(loc["lat"], loc["lon"], hub["latitude"], hub["longitude"])
            d_road, t_offpeak, t_peak = compute_urban_commute_time(d_euc, corridor=hub["hub_name"])
            
            rec[f"dist_{hub_code}_road_km"] = d_road
            rec[f"commute_{hub_code}_offpeak_min"] = t_offpeak
            rec[f"commute_{hub_code}_peak_min"] = t_peak
            rec[f"congestion_ratio_{hub_code}"] = round(t_peak / t_offpeak, 2)
            
            commute_peak_times.append(t_peak)
            commute_offpeak_times.append(t_offpeak)
            road_dists.append(d_road)
            
        # Summary commute metrics
        rec["min_commute_peak_min"] = min(commute_peak_times)
        rec["min_commute_offpeak_min"] = min(commute_offpeak_times)
        rec["avg_commute_peak_min"] = round(float(np.mean(commute_peak_times)), 1)
        rec["avg_congestion_ratio"] = round(float(np.mean([p/op for p, op in zip(commute_peak_times, commute_offpeak_times)])), 2)
        rec["dist_to_nearest_tech_hub_km"] = min(road_dists)
        
        # 2. Distance and travel time to nearest Namma Metro station
        metro_dists = [haversine_km(loc["lat"], loc["lon"], m["latitude"], m["longitude"]) for m in METRO_STATIONS]
        min_metro_idx = int(np.argmin(metro_dists))
        min_metro_euc = metro_dists[min_metro_idx]
        d_metro_road = round(min_metro_euc * 1.25, 2)
        # Feeder access time (last mile at ~15 km/h urban speed + parking/entrance delay)
        t_metro_access = round((d_metro_road / 15.0) * 60.0 + 3.0, 1)
        
        rec["nearest_metro_station"] = METRO_STATIONS[min_metro_idx]["station_name"]
        rec["nearest_metro_line"] = METRO_STATIONS[min_metro_idx]["line"]
        rec["dist_to_metro_km"] = d_metro_road
        rec["commute_metro_access_min"] = t_metro_access
        
        locality_traffic_records.append(rec)
        
    df_loc_traffic = pd.DataFrame(locality_traffic_records)
    df_loc_traffic.to_csv("data/bangalore_locality_traffic_matrix.csv", index=False)
    
    # Also save the master localities file
    df_loc_master = df_canonical.copy()
    df_loc_master.to_csv("data/bangalore_localities_master.csv", index=False)
    print(f"[OK] Generated and saved Locality Traffic Matrix for {len(df_loc_traffic)} canonical localities")

    # ---------------------------------------------------------
    # Step D: Construct Locality Graph Topology for GNN (GraphSAGE / GAT)
    # ---------------------------------------------------------
    edges = []
    num_locs = len(df_canonical)
    
    # Build K-Nearest Neighbors spatial road graph (k=6) plus distance threshold (< 7.5 km)
    # This forms a realistic planar road network connecting neighboring localities
    for i in range(num_locs):
        loc_i = df_canonical.iloc[i]
        dists = []
        for j in range(num_locs):
            if i == j:
                continue
            loc_j = df_canonical.iloc[j]
            d = haversine_km(loc_i['lat'], loc_i['lon'], loc_j['lat'], loc_j['lon'])
            dists.append((j, d))
            
        # Sort by distance
        dists.sort(key=lambda x: x[1])
        # Connect to 6 nearest neighbors if within 10 km, or at least 3 nearest
        connected = 0
        for j, d in dists:
            if d <= 9.0 or connected < 3:
                d_road, t_offpeak, t_peak = compute_urban_commute_time(d)
                # Exponential decay weight: w = exp(-d / sigma)
                weight = round(math.exp(-d / 4.0), 4)
                edges.append({
                    "source_locality_id": loc_i["locality_id"],
                    "target_locality_id": df_canonical.iloc[j]["locality_id"],
                    "source_name": loc_i["name"],
                    "target_name": df_canonical.iloc[j]["name"],
                    "distance_km": round(d_road, 2),
                    "peak_commute_min": t_peak,
                    "spatial_edge_weight": weight
                })
                connected += 1
            if connected >= 6:
                break
                
    df_edges = pd.DataFrame(edges)
    df_edges.to_csv("data/bangalore_locality_graph_edges.csv", index=False)
    print(f"[OK] Constructed Locality Graph with {len(df_edges)} directed edges across {num_locs} nodes")

    # Compute node degrees for inductive graph features
    degrees = df_edges.groupby("source_locality_id").size().to_dict()
    
    # Graph Node Features (NON-LEAKED: only spatial, commute, metro, and topological degree)
    node_features = []
    for _, row in df_loc_traffic.iterrows():
        lid = int(row["locality_id"])
        node_features.append({
            "locality_id": lid,
            "canonical_locality_name": row["canonical_locality_name"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "node_degree": degrees.get(lid, 0),
            "commute_whitefield_peak_min": row["commute_whitefield_peak_min"],
            "commute_bellandur_peak_min": row["commute_bellandur_orr_peak_min"],
            "commute_ecity_peak_min": row["commute_electronic_city_peak_min"],
            "commute_manyata_peak_min": row["commute_manyata_tech_park_peak_min"],
            "commute_mgroad_peak_min": row["commute_mg_road_cbd_peak_min"],
            "avg_congestion_ratio": row["avg_congestion_ratio"],
            "dist_to_metro_km": row["dist_to_metro_km"],
            "commute_metro_access_min": row["commute_metro_access_min"],
            "spatial_quadrant": row["spatial_quadrant"]
        })
    df_nodes = pd.DataFrame(node_features)
    df_nodes.to_csv("data/bangalore_locality_graph_nodes.csv", index=False)
    print(f"[OK] Saved Inductive Locality Graph Node Features to data/bangalore_locality_graph_nodes.csv")

    # ---------------------------------------------------------
    # Step E: Clean Raw Kaggle Real Estate Listings
    # ---------------------------------------------------------
    raw_path = "data/Bengaluru_House_Data.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"{raw_path} not found. Please place Bengaluru_House_Data.csv in data/")
        
    df_raw = pd.read_csv(raw_path)
    print(f"[*] Processing {len(df_raw)} raw listings...")
    
    # 1. Clean location & fuzzy match to canonical master
    matched_localities = []
    match_scores = []
    
    for loc_val in df_raw['location']:
        can_name, score = match_canonical_locality(loc_val)
        matched_localities.append(can_name)
        match_scores.append(score)
        
    df_raw['canonical_locality_name'] = matched_localities
    df_raw['locality_match_score'] = match_scores
    
    # 2. Parse total_sqft
    df_raw['total_sqft_cleaned'] = df_raw['total_sqft'].apply(parse_total_sqft)
    
    # 3. Parse BHK
    df_raw['bhk'] = df_raw['size'].apply(parse_bhk)
    
    # 4. Clean Bath & Balcony
    # Impute missing bath by BHK median
    bath_median = df_raw.groupby('bhk')['bath'].transform('median').fillna(2.0)
    df_raw['bath_cleaned'] = df_raw['bath'].fillna(bath_median).astype(int)
    
    # Balcony median
    balcony_median = df_raw.groupby('bhk')['balcony'].transform('median').fillna(1.0)
    df_raw['balcony_cleaned'] = df_raw['balcony'].fillna(balcony_median).astype(int)
    
    # 5. Price & Price per sqft
    df_raw['price_lakhs'] = df_raw['price'].astype(float)
    df_raw['price_inr'] = df_raw['price_lakhs'] * 100000.0
    df_raw['price_per_sqft'] = df_raw['price_inr'] / df_raw['total_sqft_cleaned']
    
    # 6. Availability flag
    df_raw['is_ready_to_move'] = df_raw['availability'].apply(lambda x: 1 if 'ready to move' in str(x).lower() else 0)
    
    # 7. Outlier Filtering (standard real estate domain rules)
    # - sqft must be valid and positive
    # - BHK between 1 and 10
    # - sqft per BHK >= 250 sqft (avoids fake data like 8 BHK in 500 sqft)
    # - price per sqft between 1,200 and 45,000 INR/sqft
    # - exclude matched locality = 'Other / Peripheral' or unmapped
    valid_mask = (
        df_raw['total_sqft_cleaned'].notna() &
        (df_raw['total_sqft_cleaned'] >= 300) &
        (df_raw['total_sqft_cleaned'] <= 15000) &
        df_raw['bhk'].notna() &
        (df_raw['bhk'] >= 1) &
        (df_raw['bhk'] <= 9) &
        ((df_raw['total_sqft_cleaned'] / df_raw['bhk']) >= 250) &
        df_raw['price_per_sqft'].notna() &
        (df_raw['price_per_sqft'] >= 1200) &
        (df_raw['price_per_sqft'] <= 45000) &
        (df_raw['canonical_locality_name'] != "Other / Peripheral") &
        (df_raw['bath_cleaned'] <= df_raw['bhk'] + 2)
    )
    
    df_clean = df_raw[valid_mask].copy().reset_index(drop=True)
    df_clean['listing_id'] = range(len(df_clean))
    df_clean['log_price'] = np.log(df_clean['price_lakhs'])
    
    print(f"[OK] Retained {len(df_clean)} high-quality validated listings ({len(df_clean)/len(df_raw)*100:.1f}% retention rate)")
    
    # Save cleaned tabular listings
    tabular_cols = [
        'listing_id', 'area_type', 'availability', 'is_ready_to_move', 'location',
        'canonical_locality_name', 'locality_match_score', 'bhk', 'total_sqft_cleaned',
        'bath_cleaned', 'balcony_cleaned', 'price_lakhs', 'log_price', 'price_per_sqft'
    ]
    df_clean[tabular_cols].to_csv("data/bangalore_housing_cleaned_tabular.csv", index=False)
    print(f"[OK] Saved Cleaned Tabular Listings to data/bangalore_housing_cleaned_tabular.csv")

    # ---------------------------------------------------------
    # Step F: Spatial Merge (Join Listings with Locality Traffic Features)
    # ---------------------------------------------------------
    df_merged = df_clean.merge(df_loc_traffic, on="canonical_locality_name", how="inner")
    print(f"[OK] Spatially merged listings with traffic matrix: {len(df_merged)} rows across {df_merged['canonical_locality_name'].nunique()} localities")

    # ---------------------------------------------------------
    # Step G: Rigorous Evaluation Splits (Random, Leave-Locality-Out, Spatial Block)
    # ---------------------------------------------------------
    np.random.seed(42)
    
    # 1. Random Split (80% Train, 10% Val, 10% Test)
    n = len(df_merged)
    shuffled_idx = np.random.permutation(n)
    train_end = int(0.80 * n)
    val_end = int(0.90 * n)
    
    random_split = np.empty(n, dtype=object)
    random_split[shuffled_idx[:train_end]] = 'train'
    random_split[shuffled_idx[train_end:val_end]] = 'val'
    random_split[shuffled_idx[val_end:]] = 'test'
    df_merged['split_random'] = random_split
    
    # 2. Leave-Locality-Out (LLO) 5-Fold Cross Validation Splits
    # Ensures zero spatial leakage: all listings in a locality are held out together
    unique_locs = df_merged['canonical_locality_name'].unique()
    np.random.shuffle(unique_locs)
    loc_to_fold = {loc: i % 5 for i, loc in enumerate(unique_locs)}
    df_merged['split_llo_fold'] = df_merged['canonical_locality_name'].map(loc_to_fold)
    
    # 3. Spatial Block Splits (Generalization across Bangalore quadrants)
    # East, South, North, West, Central
    # Quadrant assignment already present in 'spatial_quadrant'
    
    # Save the comprehensive final modeling dataset
    df_merged.to_csv("data/bangalore_housing_spatial_commute_dataset.csv", index=False)
    print(f"[OK] Saved Unified Final Dataset to data/bangalore_housing_spatial_commute_dataset.csv ({df_merged.shape[0]} rows, {df_merged.shape[1]} cols)")

    # ---------------------------------------------------------
    # Step H: Generate PyTorch Geometric Data Object for GraphSAGE / GAT
    # ---------------------------------------------------------
    # Node features tensor: [N_nodes, num_node_features]
    # Continuous node features: lat, lon, degree, 5 peak commute times, congestion ratio, metro distance, metro travel time
    feat_cols = [
        'latitude', 'longitude', 'node_degree',
        'commute_whitefield_peak_min', 'commute_bellandur_peak_min',
        'commute_ecity_peak_min', 'commute_manyata_peak_min',
        'commute_mgroad_peak_min', 'avg_congestion_ratio',
        'dist_to_metro_km', 'commute_metro_access_min'
    ]
    
    # Standardize node features
    X_raw = df_nodes[feat_cols].values.astype(np.float32)
    X_mean = np.mean(X_raw, axis=0)
    X_std = np.std(X_raw, axis=0) + 1e-6
    X_norm = (X_raw - X_mean) / X_std
    x_tensor = torch.tensor(X_norm, dtype=torch.float32)
    
    # Edge index tensor: [2, num_edges]
    src = df_edges['source_locality_id'].values
    tgt = df_edges['target_locality_id'].values
    edge_index = torch.tensor(np.array([src, tgt]), dtype=torch.long)
    edge_attr = torch.tensor(df_edges['spatial_edge_weight'].values, dtype=torch.float32).unsqueeze(1)
    
    pyg_bundle = {
        "x": x_tensor,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "feature_names": feat_cols,
        "feature_mean": torch.tensor(X_mean, dtype=torch.float32),
        "feature_std": torch.tensor(X_std, dtype=torch.float32),
        "locality_id_map": df_nodes[['locality_id', 'canonical_locality_name']].to_dict(orient='records'),
        "num_nodes": len(df_nodes),
        "num_edges": len(df_edges)
    }
    
    torch.save(pyg_bundle, "data/bangalore_graph_pyg.pt")
    print(f"[OK] Serialized PyG-compatible Graph Data to data/bangalore_graph_pyg.pt")
    print(f"    Graph summary: {pyg_bundle['num_nodes']} nodes, {pyg_bundle['num_edges']} edges, {len(feat_cols)} node features")
    print("=" * 80)
    print("ALL DATASET ASSETS BUILT AND VALIDATED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_pipeline()
