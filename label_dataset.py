import pandas as pd
import numpy as np
import random

random.seed(42)

# Load Kaggle dataset
df = pd.read_csv("Train.csv")

# ✅ Add NEW INPUT FEATURES (synthetic since Kaggle doesn't contain them)
# These simulate real-world packaging needs
df["Product_category"] = np.random.choice(
    ["food", "electronics", "cosmetics", "clothing", "books", "glass_items"], size=len(df)
)

df["Length_cm"] = np.random.randint(5, 60, size=len(df))
df["Width_cm"] = np.random.randint(5, 50, size=len(df))
df["Height_cm"] = np.random.randint(2, 40, size=len(df))

df["Fragile"] = np.random.choice([0, 1], size=len(df))
df["Liquid_Leak_Prone"] = np.random.choice([0, 1], size=len(df))
df["Perishable"] = np.random.choice([0, 1], size=len(df))

df["Shipping_distance"] = np.random.choice(["local", "domestic", "international"], size=len(df))
df["Stacking_pressure"] = np.random.choice([0, 1], size=len(df))
df["Delivery_urgency"] = np.random.choice(["standard", "express"], size=len(df))

df["Budget_level"] = np.random.choice(["low", "medium", "high"], size=len(df))
df["Eco_preference"] = np.random.choice(["max_sustainability", "balanced", "low_cost"], size=len(df))
df["Material_constraint"] = np.random.choice(
    ["no_plastic", "only_recyclable", "compostable_only", "no_restrictions"], size=len(df)
)

df["Return_risk"] = np.random.choice([0, 1], size=len(df))


# ✅ Packaging label logic based on all features
def assign_packaging(row):
    weight = row["Weight_in_gms"]
    fragile = row["Fragile"]
    liquid = row["Liquid_Leak_Prone"]
    perishable = row["Perishable"]
    distance = row["Shipping_distance"]
    urgency = row["Delivery_urgency"]
    budget = row["Budget_level"]
    eco_pref = row["Eco_preference"]
    constraint = row["Material_constraint"]
    importance = row["Product_importance"]
    cost = row["Cost_of_the_Product"]

    # ✅ Rule Overrides for safety
    if liquid == 1:
        return "Leak-proof Compostable Container"

    if perishable == 1:
        return "Insulated Compostable Food Pack"

    if fragile == 1 and distance == "international":
        return "Molded Pulp Tray + Corrugated Box"

    # ✅ General weight-based recommendation
    if weight <= 500:
        pkg = "Biodegradable Mailer"
    elif weight <= 1500:
        pkg = "Recycled Cardboard Box"
    elif weight <= 3000:
        pkg = "Corrugated Box + Paper Cushion"
    else:
        pkg = "Corrugated Box + Molded Pulp"

    # ✅ Importance upgrade
    if importance == "high" or cost > 250:
        pkg = "Molded Pulp Tray + Corrugated Box"

    # ✅ Eco preference changes
    if eco_pref == "max_sustainability":
        if pkg in ["Corrugated Box + Molded Pulp", "Molded Pulp Tray + Corrugated Box"]:
            return "Molded Pulp Tray + Corrugated Box"
        return "Recycled Cardboard Box"

    if eco_pref == "low_cost" and budget == "low":
        return "Recycled Cardboard Box"

    # ✅ Material constraints filtering
    if constraint == "compostable_only":
        return "Compostable Paper Wrap Pack"

    if constraint == "only_recyclable":
        return "100% Recyclable Cardboard Box"

    if constraint == "no_plastic":
        return pkg  # already plastic-free options

    # ✅ Express urgency upgrade
    if urgency == "express" and fragile == 1:
        return "Molded Pulp Tray + Corrugated Box"

    return pkg


df["recommended_packaging"] = df.apply(assign_packaging, axis=1)

# Save new dataset
df.to_csv("ecopack_kaggle_labeled.csv", index=False)

print("✅ Labeled dataset saved: ecopack_kaggle_labeled.csv")
print(df[[
    "Product_category", "Weight_in_gms", "Fragile", "Liquid_Leak_Prone", "Perishable",
    "Shipping_distance", "Delivery_urgency", "Budget_level", "Eco_preference",
    "Material_constraint", "Product_importance", "recommended_packaging"
]].head())
