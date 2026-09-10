# Data Sources, Methodology & Governance

## 1. Primary Dataset: Indian Quick-Commerce & Food Delivery Logistics
* **Origin**: Swiggy & Zomato Delivery Partner Telemetry and Order Operations Dataset.
* **Scope**: 45,502 real customer delivery orders across 22 major Indian metropolitan and urban hubs.
* **Time Period**: February 11, 2022 to April 6, 2022 (capturing real-world peak traffic, monsoon/summer storms, and evening rush hours).
* **Format**: Comma-Separated Values (CSV), 8.08 MB raw file.

## 2. Metropolitan Hubs Covered

| City Code | Metro Hub | Typical Operational Bottleneck |
| :--- | :--- | :--- |
| `BANG` | Bangalore | Tech corridor congestion (Silk Board, Outer Ring Road, Marathahalli, Whitefield) |
| `DEL` | Delhi NCR | Long delivery radii, dense urban bottlenecks (Cyber Hub, Connaught Place, Noida) |
| `MUM` | Mumbai | Monsoon storms, heavy highway congestion (Western Express Highway, Andheri, BKC) |
| `HYD` | Hyderabad | Hitec City & Gachibowli evening dinner rush hours |
| `PUNE` | Pune | Hinjewadi & Viman Nagar IT corridor spikes |
| `CHEN` | Chennai | Coastal weather disruptions, high humidity |
| `KOL` | Kolkata | High density narrow urban delivery routes |
| `JAI` / `INDO` | Jaipur / Indore | Rapidly growing Tier-1/2 quick-commerce expansion |

## 3. Feature Schema & Operational Role

| Column | Type | Operational Role |
| :--- | :--- | :--- |
| `rider_id` | String | Unique delivery partner alphanumeric ID |
| `age` | Float | Delivery partner age (experience baseline) |
| `ratings` | Float | Delivery partner customer rating (speed & service quality) |
| `restaurant_latitude` / `_longitude` | Float | Kitchen pickup GPS coordinates |
| `delivery_latitude` / `_longitude` | Float | Customer drop-off GPS coordinates |
| `distance` | Float | Haversine trip distance in kilometers (clipped 0.5 to 25 km) |
| `order_date` | Date | Order date (February–April 2022) |
| `weather` | Categorical | `sunny`, `stormy`, `rainy`, `cloudy`, `fog`, `sandstorms` |
| `traffic` | Categorical | `jam`, `high`, `medium`, `low` |
| `vehicle_condition` | Integer | Mechanical condition of bike/scooter (0, 1, 2) |
| `type_of_vehicle` | Categorical | `motorcycle`, `scooter`, `electric_bike`, `bicycle` |
| `multiple_deliveries` | Integer | Number of stacked/batched orders on a single trip (0, 1, 2, 3) |
| `festival` | Categorical | Festival surge indicator (`yes` / `no`) |
| `city_type` | Categorical | `metropolitian`, `urban`, `semi-urban` |
| `city_name` | Categorical | Indian city identifier code (`BANG`, `DEL`, `MUM`, etc.) |
| `pickup_time_minutes` | Float | Food preparation and kitchen dispatch delay (minutes) |
| `order_time_hour` | Float | Hour of day (0–23) |
| `time_taken` | Float | **Target Variable**: Actual total delivery duration in minutes |

## 4. Leakage Prevention & Preprocessing Standards

1. **Strict 80/20 Partitioning**: Train split ($N = 36,401$) and test split ($N = 9,101$) were partitioned before calculating any scaling or frequency metrics.
2. **Imputation Rules**: Numerical features (rider ratings, age, kitchen prep delays) use median imputation; categoricals use mode imputation.
3. **Encapsulation**: All transformations are serialized inside a scikit-learn `ColumnTransformer` to ensure reproducibility.
