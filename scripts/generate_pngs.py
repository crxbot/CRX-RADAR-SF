import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as path_effects
import matplotlib.patches as mpatches
from pyproj import Transformer, CRS
import pandas as pd
import os
import glob
from shapely.geometry import Polygon
from shapely.ops import unary_union
import shapely.vectorized as sv

# ======================
# Verzeichnisse
# ======================
INPUT_DIR = "data/radar"
OUTPUT_DIR = "output/radar"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# Projektionsparameter (DWD RADOLAN, verifiziert)
# ======================
PROJ_STRING = (
    "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=10 "
    "+a=6370040 +b=6370040 +no_defs "
    "+x_0=542962.16692185658 +y_0=3609144.7242655745"
)

# ======================
# Städte
# ======================
cities = pd.DataFrame({
    'name': ['Berlin', 'Hamburg', 'München', 'Köln', 'Frankfurt', 'Dresden',
             'Stuttgart', 'Düsseldorf', 'Nürnberg', 'Erfurt', 'Leipzig',
             'Bremen', 'Saarbrücken', 'Hannover'],
    'lat': [52.52, 53.55, 48.14, 50.94, 50.11, 51.05,
            48.78, 51.23, 49.45, 50.98, 51.34, 53.08, 49.24, 52.37],
    'lon': [13.40, 9.99, 11.57, 6.96, 8.68, 13.73,
            9.18, 6.78, 11.08, 11.03, 12.37, 8.80, 6.99, 9.73]
})


# ======================
# Datei finden
# ======================
rw_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.hdf5")))
if not rw_files:
    raise RuntimeError("Keine RW-HDF5-Datei gefunden!")

last_file = rw_files[-1]
print("Processing:", last_file)


def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        pass


with h5py.File(last_file, "r") as h5file:
    h5file.visititems(print_structure)

# ======================
# ODIM-HDF5 korrekt & robust lesen
# ======================
with h5py.File(last_file, "r") as h5file:
    # Prüfe ob Quality-Daten vorhanden sind
    print("\n=== QUALITY CHECK ===")
    if "dataset1/quality1" in h5file:
        if "data" in h5file["dataset1/quality1"]:
            quality = h5file["dataset1/quality1/data"][()]
    else:
        quality = None
    
    where = h5file["where"]

    # ---- Produkt-Zeit finden (robust) ----
    what_prod = h5file["what"]

    # HDF5 Dataset
    raw = h5file["dataset1/data1/data"][()]

    # ---- Skalierung ----
    what_data = h5file["dataset1/data1/what"]
    gain = float(what_data.attrs["gain"])
    offset = float(what_data.attrs["offset"])
    nodata = int(what_data.attrs["nodata"])
    undetect = int(what_data.attrs["undetect"])

    nodata_pixels = np.sum(raw == nodata)
    undetect_pixels = np.sum(raw == undetect)

    WIDTH = int(where.attrs["xsize"])
    HEIGHT = int(where.attrs["ysize"])

    # ---- Zeit ----
    date = what_prod.attrs["date"].decode("ascii")
    time = what_prod.attrs["time"].decode("ascii")

# ======================
# DATENVERARBEITUNG - KORREKTUR
# ======================

# 1. Rohdaten in float umwandeln
data = raw.astype(float)

# 2. Maske für echte Fehlwerte (nodata) erstellen
# Diese werden später durch cmap.set_bad() grau
nodata_mask = (raw == nodata)

# 3. Undetect (trocken) auf einen Wert setzen, der sicher unter 0.1 liegt
# Wir setzen es auf einen speziellen Wert, z.B. -1.0, 
# damit es von der Norm als "under" erkannt wird.
data[raw == undetect] = -1.0

# 4. Skalierung NUR auf echte Messwerte anwenden
# (Dort wo nicht nodata und nicht undetect ist)
valid_mask = (raw != nodata) & (raw != undetect)
data[valid_mask] = data[valid_mask] * gain + offset

# 5. Maskierte Array erstellen für nodata
data = np.ma.array(data, mask=nodata_mask)

# ======================
# Zeitstempel
# ======================
utc_time = pd.to_datetime(date + time, format="%Y%m%d%H%M%S", utc=True)
local_time = utc_time.tz_convert("Europe/Berlin")
footer_time_str = local_time.strftime("%d.%m.%Y %H:%M")

output_filename = f"radolan_sf_{local_time.strftime('%Y%m%d_%H%M')}.png"
output_path = os.path.join(OUTPUT_DIR, output_filename)

# ======================
# Gitter (Pixelmittelpunkte, DWD-konform)
# ======================
x_edges = (np.arange(WIDTH + 1) - 0.5) * 1000.0
y_edges = -(np.arange(HEIGHT + 1) - 0.5) * 1000.0
xg, yg = np.meshgrid(x_edges, y_edges)

crs_proj = CRS.from_proj4(PROJ_STRING)
crs_wgs84 = CRS.from_epsg(4326)
transformer = Transformer.from_crs(crs_proj, crs_wgs84, always_xy=True)

lon_grid, lat_grid = transformer.transform(xg, yg)

# ======================
# RADOLAN-Abdeckung (KORRIGIERT nach wradlib)
# ======================

# Radar-Standorte (lon/lat)
radars_dict = dict(
    asb=dict(name="ASR Borkum", lat=53.564011, lon=6.748292),
    boo=dict(name="Boostedt", lat=54.00438, lon=10.04687),
    drs=dict(name="Dresden", lat=51.12465, lon=13.76865),
    eis=dict(name="Eisberg", lat=49.54066, lon=12.40278),
    emd=dict(name="Emden", lat=53.33872, lon=7.02377),
    ess=dict(name="Essen", lat=51.40563, lon=6.96712),
    fbg=dict(name="Feldberg", lat=47.87361, lon=8.00361),
    fld=dict(name="Flechtdorf", lat=51.31120, lon=8.802),
    hnr=dict(name="Hannover", lat=52.46008, lon=9.69452),
    neu=dict(name="Neuhaus", lat=50.50012, lon=11.13504),
    nhb=dict(name="Neuheilenbach", lat=50.10965, lon=6.54853),
    oft=dict(name="Offenthal", lat=49.9847, lon=8.71293),
    pro=dict(name="Prötzel", lat=52.64867, lon=13.85821),
    mem=dict(name="Memmingen", lat=48.04214, lon=10.21924),
    ros=dict(name="Rostock", lat=54.17566, lon=12.05808),
    isn=dict(name="Isen", lat=48.17470, lon=12.10177),
    tur=dict(name="Türkheim", lat=48.58528, lon=9.78278),
    umd=dict(name="Ummendorf", lat=52.16009, lon=11.17609),
)

MAX_RANGE_KM = 150

def create_radar_coverage_polar(radar_lon, radar_lat, max_range_km, crs_proj, n_az=360, n_range=150):
    """
    Erstellt die Abdeckung eines Radars durch polare Koordinaten in der korrekten Projektion.
    Dies berücksichtigt die Erdkrümmung und die stereografische Projektion korrekt.
    """
    # Transformiere Radar-Position nach Projektion
    transformer_to_proj = Transformer.from_crs(crs_wgs84, crs_proj, always_xy=True)
    radar_x, radar_y = transformer_to_proj.transform(radar_lon, radar_lat)
    
    # Erstelle polares Gitter (wie bei echten Radardaten)
    azimuths = np.linspace(0, 360, n_az, endpoint=False)
    ranges = np.linspace(0, max_range_km * 1000, n_range)  # in Metern
    
    # Berechne kartesische Koordinaten in der Projektion
    az_rad = np.deg2rad(azimuths)
    
    # Äußerer Ring des Radarbereichs
    outer_x = radar_x + ranges[-1] * np.sin(az_rad)
    outer_y = radar_y + ranges[-1] * np.cos(az_rad)
    
    # Transformiere zurück nach WGS84
    transformer_to_wgs = Transformer.from_crs(crs_proj, crs_wgs84, always_xy=True)
    outer_lons, outer_lats = transformer_to_wgs.transform(outer_x, outer_y)
    
    # Erstelle Polygon
    coords = np.column_stack([outer_lons, outer_lats])
    return Polygon(coords)


print("\n=== Erstelle RADOLAN-Abdeckungsmaske ===")
coverage_polygons = []

for radar_id, radar_info in radars_dict.items():
    poly = create_radar_coverage_polar(
        radar_info["lon"], 
        radar_info["lat"], 
        MAX_RANGE_KM,
        crs_proj
    )
    coverage_polygons.append(poly)
    print(f"  {radar_id}: {radar_info['name']}")

# Vereinige alle Radar-Abdeckungen
radolan_coverage = unary_union(coverage_polygons)

# Berechne Pixelmittelpunkte
lon_centers = (lon_grid[:-1, :-1] + lon_grid[1:, 1:]) / 2
lat_centers = (lat_grid[:-1, :-1] + lat_grid[1:, 1:]) / 2

# Erstelle Maske (Pixel außerhalb der Abdeckung)
print("Berechne Abdeckungsmaske...")
outside_mask = ~sv.contains(radolan_coverage, lon_centers, lat_centers)

# Wende Maske an
data = np.ma.array(data, mask=(data.mask | outside_mask))

pixels_masked = np.sum(outside_mask)
print(f"Pixel außerhalb RADOLAN-Abdeckung: {pixels_masked}")


# ======================
# Farben / Levels
# ======================
colors = [
    "#00C9FF", "#0057FF", "#0000EE", "#BEFFBD", "#98FE98", "#69FF68",
    "#30FF30", "#0AFF0A", "#00DC00", "#00BF00", "#008D00", "#FFFF00",
    "#F1D801", "#EABA00", "#F99C00", "#FE4100", "#FF2700", "#DC0000",
    "#B00000", "#FAC3FC", "#EBAAEA", "#DD95DE", "#C674C6", "#BA62B9",
    "#A342A3", "#861686", "#5C0F5C", "#410A41", "#320732"
]
levels = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30,
          35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 130, 150, 250, 350]

cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N)

# Das sind die "undetect" Bereiche (trocken) -> Dunkleres Grau
cmap.set_under("#676767") 

# Das sind die "nodata" Bereiche (Lücken/Fehler) -> Helleres Grau
cmap.set_bad("#909090")


# ======================
# Plot
# ======================
FIG_W_PX, FIG_H_PX = 880, 830
BOTTOM_AREA_PX = 179
TOP_AREA_PX = FIG_H_PX - BOTTOM_AREA_PX
TARGET_ASPECT = FIG_W_PX / TOP_AREA_PX

extent = [5, 16, 47, 56]

scale = 0.9
fig = plt.figure(figsize=(FIG_W_PX/100*scale, FIG_H_PX/100*scale), dpi=100)
shift_up = 0.02
ax = fig.add_axes([0.0, BOTTOM_AREA_PX / FIG_H_PX + shift_up, 1.0, TOP_AREA_PX / FIG_H_PX],
                    projection=ccrs.PlateCarree())
ax.set_extent(extent)
ax.set_axis_off()
ax.set_aspect('auto')

ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor="#2C2C2C", linewidth=1)
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.add_feature(cfeature.COASTLINE)

for _, city in cities.iterrows():
    ax.plot(city["lon"], city["lat"], "o", markersize=6,
            markerfacecolor="black", markeredgecolor="white",
            markeredgewidth=1.5, zorder=5)
    txt = ax.text(city["lon"]+0.1, city["lat"]+0.1, city["name"],
                  fontsize=9, color="black", weight="bold", zorder=6)
    txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="white")])

ax.add_patch(
    mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                       fill=False, color="black", linewidth=2)
)

im = ax.pcolormesh(
    lon_grid, lat_grid, data,
    cmap=cmap, norm=norm, shading="auto",
    transform=ccrs.PlateCarree()
)

# ======================
# Colorbar
# ======================
cbar_ax = fig.add_axes([0.03, 45 / FIG_H_PX, 0.94, 50 / FIG_H_PX])
cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", ticks=levels)
cbar.ax.tick_params(labelsize=7)
cbar.outline.set_edgecolor("black")
cbar.ax.set_facecolor("#676767")
cbar.set_ticklabels([str(int(v)) if float(v).is_integer() else str(v) for v in levels])

# ======================
# Footer
# ======================
footer_ax = fig.add_axes(
    [0.0, (45 + 50) / FIG_H_PX, 1.0,
     (BOTTOM_AREA_PX - 50 - 45) / FIG_H_PX]
)
footer_ax.axis("off")

footer_ax.text(0.01, 0.85, "Niederschlag, 24Std (mm)",
               fontsize=12, fontweight="bold", ha="left", va="top")
footer_ax.text(0.01, 0.55, "Daten: Deutscher Wetterdienst",
               fontsize=8, ha="left", va="top")
footer_ax.text(0.99, 0.85, footer_time_str + " Uhr",
               fontsize=12, fontweight="bold", ha="right", va="top")

# ======================
# Speichern
# ======================
plt.savefig(output_path, dpi=100, pad_inches=0)
plt.close()

print("\n✓ Saved:", output_path)
