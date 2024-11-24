#!/usr/bin/env python3
"""
Fetch a list of planets habitated by sentient species
"""
import requests


def sentientPlanets():
    """
    Fetch a list of homeworld planets of all sentient species.
    Maintains the order provided by the SWAPI API.
    """
    species_url = "https://swapi-api.hbtn.io/api/species/"
    planets_list = []

    while species_url:
        response = requests.get(species_url)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch data: {response.status_code}")

        data = response.json()

        for species in data.get("results", []):
            # Check if species is sentient
            classification = species.get("classification", "").lower()
            designation = species.get("designation", "").lower()
            if "sentient" in classification or "sentient" in designation:
                # Fetch homeworld name
                homeworld_url = species.get("homeworld")
                if homeworld_url:
                    homeworld_response = requests.get(homeworld_url)
                    if homeworld_response.status_code == 200:
                        homeworld_name = homeworld_response.json().get("name")
                        # Only add if name exists
                        if homeworld_name:
                            planets_list.append(homeworld_name)

        # Get the next page
        species_url = data.get("next")

    return planets_list
