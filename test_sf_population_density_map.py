#!/usr/bin/env -S uv run --with pandas --with geopandas --with folium
from __future__ import annotations

import json
import math
import pathlib
import urllib.parse
import urllib.request

import folium
import geopandas as gpd
import pandas as pd

TRACT_GEOJSON_URL = "https://data.sfgov.org/resource/tmph-tgz9.geojson?$limit=50000"
DEMOGRAPHICS_API_URL = "https://data.sfgov.org/resource/4qbq-hvtt.json"


def fetch_json(url: str, params: dict[str, str] | None = None) -> list[dict]:
    if params:
        query = urllib.parse.urlencode(params)
        url = f"{url}?{query}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def load_tract_geometries() -> gpd.GeoDataFrame:
    tracts = gpd.read_file(TRACT_GEOJSON_URL)
    tracts["geoid"] = tracts["geoid"].astype(str)
    return tracts[["geoid", "geometry"]].copy()


def load_population_totals() -> pd.DataFrame:
    rows = fetch_json(
        DEMOGRAPHICS_API_URL,
        {
            "$select": "geography_id,estimate,start_year,end_year",
            "$where": (
                "geography='tract' "
                "AND acs_table='B01001' "
                "AND acs_label='Estimate!!Total:' "
                "AND start_year=2019 AND end_year=2023"
            ),
            "$limit": "50000",
        },
    )
    pop = pd.DataFrame(rows)
    if pop.empty:
        raise RuntimeError("No tract-level population rows returned from 4qbq-hvtt")

    pop["geography_id"] = pop["geography_id"].astype(str)
    pop["population"] = pd.to_numeric(pop["estimate"], errors="coerce")
    pop = pop.dropna(subset=["population"])
    pop["population"] = pop["population"].astype(int)
    return pop[["geography_id", "population", "start_year", "end_year"]].drop_duplicates()


def compute_density_map_data(tracts: gpd.GeoDataFrame, population: pd.DataFrame) -> gpd.GeoDataFrame:
    merged = tracts.merge(population, left_on="geoid", right_on="geography_id", how="left")
    merged = merged.dropna(subset=["population"]).copy()

    area_projected = merged.to_crs("EPSG:3310")
    merged["area_sq_km"] = area_projected.geometry.area / 1_000_000.0
    merged["pop_density_per_sq_km"] = merged["population"] / merged["area_sq_km"]
    merged["pop_density_sqrt"] = merged["pop_density_per_sq_km"].apply(lambda x: math.sqrt(x + 1.0))
    return merged


def build_choropleth(gdf: gpd.GeoDataFrame, output_html: pathlib.Path) -> None:
    center = gdf.to_crs("EPSG:4326").geometry.union_all().centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=12, tiles="CartoDB positron")

    folium.Choropleth(
        geo_data=gdf.to_json(),
        data=gdf,
        columns=["geoid", "pop_density_sqrt"],
        key_on="feature.properties.geoid",
        fill_color="YlOrRd",
        fill_opacity=0.75,
        line_opacity=0.35,
        legend_name="Sqrt(Population Density + 1), density in people per sq km (ACS 2019-2023)",
    ).add_to(m)

    tooltip = folium.GeoJson(
        gdf.to_json(),
        style_function=lambda _: {"color": "#333", "weight": 0.5, "fillOpacity": 0},
        tooltip=folium.GeoJsonTooltip(
            fields=["geoid", "population", "area_sq_km", "pop_density_per_sq_km", "pop_density_sqrt"],
            aliases=[
                "Tract GEOID",
                "Population",
                "Area (sq km)",
                "Density (per sq km)",
                "Sqrt(Density + 1)",
            ],
            localize=True,
            sticky=False,
        ),
    )
    tooltip.add_to(m)

    m.save(output_html)


def main() -> None:
    tracts = load_tract_geometries()
    population = load_population_totals()
    gdf = compute_density_map_data(tracts, population)

    output_dir = pathlib.Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    out_csv = output_dir / "sf_population_density_tracts_2019_2023.csv"
    out_html = pathlib.Path("sf_population_density_map.html")

    gdf.drop(columns=["geometry"]).to_csv(out_csv, index=False)
    build_choropleth(gdf, out_html)

    print(f"Joined tracts: {len(gdf)}")
    print(f"Wrote density table: {out_csv.resolve()}")
    print(f"Wrote density map: {out_html.resolve()}")


if __name__ == "__main__":
    main()
