# Import the pandas library once at the top
import pandas as pd
import networkx as nx

# --- PHASE 1: SETUP AND DATA PREPARATION ---
print("--- Phase 1: Loading and cleaning primary dataset ---")

# Step 2: Load and Clean the Primary Dataset (companion_plants.csv)
try:
    companion_plants_df = pd.read_csv("companion_plants.csv")
    print("Successfully loaded companion_plants.csv")
except FileNotFoundError:
    print("Error: 'companion_plants.csv' not found. Please ensure it's in the same folder as your script.")
    exit()

# A single, clean dictionary for all name replacements
name_replacements = {
    "amaranthus cruentus": "amaranth", "chamomille": "chamomile", "apple": "apples",
    "summer savoury": "summer savory", "angelica archangelica": "angelica", "aubergine": "eggplant",
    "bee balm monarda": "bee balm", "black walnut": "walnut", "cabbages": "cabbage",
    "beetroot": "beets", "capsicum": "peppers", "chards": "chard", "corier": "coriander",
    "cucumber": "cucumbers", "delion": "dandelion", "grain sorghum": "sorghum",
    "leek": "leeks", "malting barley": "barley", "marigold": "marigolds", "melon": "melons",
    "nasturtium": "nasturtiums", "onion": "onions", "petunia": "petunias",
    "potato": "potatoes", "pot marigold": "calendula", "pumpkin": "pumpkins",
    "radish": "radishes", "rose": "roses", "soybean": "soybeans", "sunflower": "sunflowers",
    "sweet alyssum": "alyssum", "sweet potato": "sweet potatoes", "swiss chard": "chard",
    "bush": "beans, bush", "pole": "beans, pole", "rape": "rapeseed",
    "redroot pigweed": "pigweed", "lady phacelia": "phacelia"
} #

companion_plants_df.replace(name_replacements, inplace=True)
companion_plants_df.drop_duplicates(inplace=True)
companion_plants_df = companion_plants_df[companion_plants_df['Source Node'] != companion_plants_df['Destination Node']]

nodes_to_remove = ['almost everything', '\"', 'relatives', 'runner']
companion_plants_df = companion_plants_df[~companion_plants_df['Destination Node'].isin(nodes_to_remove)]


# --- PHASE 2: DATA EXTRACTION AND INTEGRATION ---
print("\n--- Phase 2: Extracting data from descriptive CSV ---")

# Step 3: Load and Process the Descriptive Dataset (companion_plants_veg.csv)
try:
    veg_df = pd.read_csv("companion_plants_veg.csv")
    print("Successfully loaded companion_plants_veg.csv")
except FileNotFoundError:
    print("Error: 'companion_plants_veg.csv' not found. Please ensure it's in the same folder as your script.")
    exit()

new_relationships = []
link_columns = {'Helps': 'helps', 'Helped by': 'helped_by', 'Avoid': 'avoid'}

for index, row in veg_df.iterrows():
    source_plant = row['Common name']
    if isinstance(source_plant, str):
        for col_name, link_type in link_columns.items():
            if isinstance(row[col_name], str):
                destination_plants = row[col_name].split(',')
                for dest_plant in destination_plants:
                    dest_plant_cleaned = dest_plant.strip()
                    if dest_plant_cleaned:
                        new_relationships.append([source_plant, link_type, dest_plant_cleaned])

extracted_df = pd.DataFrame(new_relationships, columns=['Source Node', 'Link', 'Destination Node'])
extracted_df.replace(name_replacements, inplace=True)


# --- Step 4: Combine and Finalize the Master Dataset ---
print("\n--- Step 4: Combining the two datasets ---")
master_df = pd.concat([companion_plants_df, extracted_df], ignore_index=True)
master_df['Source Node'] = master_df['Source Node'].str.lower()
master_df['Destination Node'] = master_df['Destination Node'].str.lower()
master_df.drop_duplicates(inplace=True)


# --- PHASE 3: NETWORK CREATION AND ANALYSIS ---
print("\n--- Phase 3: Creating 'Help' and 'Avoid' Networks ---")

# --- Step 5: Create the specialized networks ---
helps_df = master_df[master_df['Link'] == 'helps']
helped_by_df = master_df[master_df['Link'] == 'helped_by']
helped_by_reversed_df = helped_by_df.rename(columns={'Source Node': 'Destination Node', 'Destination Node': 'Source Node'})
help_network_df = pd.concat([
    helps_df[['Source Node', 'Destination Node']],
    helped_by_reversed_df[['Source Node', 'Destination Node']]
], ignore_index=True)
help_network_df.drop_duplicates(inplace=True)

avoid_network_df = master_df[master_df['Link'] == 'avoid'][['Source Node', 'Destination Node']]
avoid_network_df.drop_duplicates(inplace=True)

# --- NEW: DEBUGGING CHECK TO FIND THE EXACT PROBLEM ---
print("\n--- DEBUGGING: Checking for invalid data types before creating graph ---")
error_found = False
for index, row in help_network_df.iterrows():
    # Check if either the source or destination is not a string
    if not isinstance(row['Source Node'], str) or not isinstance(row['Destination Node'], str):
        print(f"!!! PROBLEM FOUND in 'help_network_df' at index {index} !!!")
        print(f"Source: {row['Source Node']} (Type: {type(row['Source Node'])})")
        print(f"Destination: {row['Destination Node']} (Type: {type(row['Destination Node'])})")
        error_found = True
        break

if error_found:
    print("\nAborting script due to invalid data type found above.")
    exit()
else:
    print("Data type check passed successfully.")


# --- Step 6: Apply the PageRank Algorithm ---
print("\n--- Step 6: Calculating PageRank scores ---")
help_graph = nx.DiGraph(help_network_df.values.tolist())
avoid_graph = nx.Graph(avoid_network_df.values.tolist())

help_ranks = nx.pagerank(help_graph, alpha=0.85)
avoid_ranks = nx.pagerank(avoid_graph, alpha=0.85)

help_ranks_df = pd.DataFrame(list(help_ranks.items()), columns=['Plant', 'Help_Rank']).sort_values(by='Help_Rank', ascending=False)
avoid_ranks_df = pd.DataFrame(list(avoid_ranks.items()), columns=['Plant', 'Avoid_Rank']).sort_values(by='Avoid_Rank', ascending=False)

print("\nPageRank calculation complete.")
print("\nTop 5 Most Beneficial Plants (by Help_Rank):")
print(help_ranks_df.head())
print("\nTop 5 Most Antagonistic Plants (by Avoid_Rank):")
print(avoid_ranks_df.head())


# --- PHASE 4: OUTPUT AND VISUALIZATION ---
print("\n--- Phase 4: Generating final outputs and visualizations ---")

# --- Step 7: Save final rankings and create bar chart ---
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Merge the help and avoid ranks into a single dataframe
# We use an outer merge to keep all plants, even if they only have one type of rank
final_ranks_df = pd.merge(help_ranks_df, avoid_ranks_df, on='Plant', how='outer')

# 2. Save the final rankings to a CSV file
final_ranks_df.to_csv('final_plant_rankings.csv', index=False)
print("\nFinal ranked data has been saved to 'final_plant_rankings.csv'")


# 3. Create a visualization of the Top 10 Most Beneficial Plants
# Set the style and size for the plot
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

# Create the bar plot
barplot = sns.barplot(
    x='Help_Rank',
    y='Plant',
    data=help_ranks_df.head(10),
    palette='viridis'
)

# Set the title and labels for clarity
plt.title('Top 10 Most Beneficial Plants (by Help Rank)', fontsize=16)
plt.xlabel('Help Rank (Higher is Better)', fontsize=12)
plt.ylabel('Plant', fontsize=12)

# Display the plot
print("\nDisplaying visualization...")
plt.tight_layout()
plt.show()



# Add these lines at the end of analysis.py
help_network_df.to_csv('help_network.csv', index=False)
avoid_network_df.to_csv('avoid_network.csv', index=False)