def generate_municipal_plan(city_name, context):
    """
    Generates a localized 3-phase Municipal Action Plan based on the city context.
    """
    city_name = city_name.capitalize()
    
    plan = f"""
### 📋 Future Urban Action Plan: {city_name}

Based on the climate profile ({context['climate']}) and current challenges ({context['issue']}), the following 3-phase strategic plan is recommended for the {city_name} Municipal Corporation:

#### Phase 1: Immediate Intervention (0-2 Years)
*   **De-Paving Initiative**: Identify highly paved public squares and obsolete parking lots. Convert at least 15% of these surfaces to permeable materials.
*   **Cool Roofs Program**: Mandate white or reflective paint on commercial buildings over 500 sq meters to combat the summer peak of {context['temp']}.

#### Phase 2: Native Afforestation Strategy (2-5 Years)
*   **Targeted Canopy Expansion**: Focus on planting native, drought-resistant shade trees such as **{context['trees']}** along major transit corridors.
*   **Green Corridors**: Connect existing fragmented maidans and parks to create continuous green corridors, allowing natural ventilation pathways.

#### Phase 3: Long-Term Sustainable Infrastructure (5-10 Years)
*   **Policy Integration**: Integrate "Cool City" mandates into future zoning laws, requiring new developments to maintain a minimum of 30% green cover (NDVI > 0.4).
*   **Water-Sensitive Urban Design**: Rejuvenate local water bodies and wetlands to naturally lower ambient temperatures during intense heatwaves.
"""
    return plan
